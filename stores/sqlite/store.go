package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/promptrails/memoryrails"
)

// Store implements memoryrails.Store using SQLite with in-process
// cosine similarity computation. Works without sqlite-vec extension
// by storing embeddings as JSON arrays and computing similarity in Go.
type Store struct {
	db *sql.DB
}

// New creates a new SQLite store. Creates the memories table if it doesn't exist.
func New(db *sql.DB) (*Store, error) {
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS memories (
			id TEXT PRIMARY KEY,
			content TEXT NOT NULL,
			type TEXT NOT NULL,
			embedding TEXT,
			metadata TEXT,
			importance REAL DEFAULT 0.5,
			access_count INTEGER DEFAULT 0,
			last_accessed_at TEXT,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)
	`)
	if err != nil {
		return fmt.Errorf("sqlite store: migration failed: %w", err)
	}

	_, _ = s.db.Exec("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
	_, _ = s.db.Exec("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
	return nil
}

func (s *Store) Put(ctx context.Context, mem *memoryrails.Memory) error {
	embJSON, _ := json.Marshal(mem.Embedding)
	metaJSON, _ := json.Marshal(mem.Metadata)

	var lastAccessed *string
	if !mem.LastAccessedAt.IsZero() {
		t := mem.LastAccessedAt.Format(time.RFC3339)
		lastAccessed = &t
	}

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO memories (id, content, type, embedding, metadata, importance, access_count, last_accessed_at, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			content=excluded.content, type=excluded.type, embedding=excluded.embedding,
			metadata=excluded.metadata, importance=excluded.importance,
			access_count=excluded.access_count, last_accessed_at=excluded.last_accessed_at,
			updated_at=excluded.updated_at
	`, mem.ID, mem.Content, string(mem.Type), string(embJSON), string(metaJSON),
		mem.Importance, mem.AccessCount, lastAccessed,
		mem.CreatedAt.Format(time.RFC3339), mem.UpdatedAt.Format(time.RFC3339))

	if err != nil {
		return fmt.Errorf("sqlite store: put failed: %w", err)
	}
	return nil
}

func (s *Store) Get(ctx context.Context, id string) (*memoryrails.Memory, error) {
	row := s.db.QueryRowContext(ctx, "SELECT id, content, type, embedding, metadata, importance, access_count, last_accessed_at, created_at, updated_at FROM memories WHERE id = ?", id)
	return s.scanRow(row)
}

func (s *Store) Search(ctx context.Context, embedding []float32, opts memoryrails.SearchOptions) ([]memoryrails.SearchResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	// Fetch all memories and compute similarity in Go
	query := "SELECT id, content, type, embedding, metadata, importance, access_count, last_accessed_at, created_at, updated_at FROM memories WHERE 1=1"
	var args []interface{}

	if opts.Type != "" {
		query += " AND type = ?"
		args = append(args, string(opts.Type))
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: search query failed: %w", err)
	}
	defer rows.Close()

	var results []memoryrails.SearchResult
	for rows.Next() {
		mem, err := s.scanRows(rows)
		if err != nil || mem == nil || len(mem.Embedding) == 0 {
			continue
		}

		sim := cosineSimilarity(embedding, mem.Embedding)
		if sim >= threshold {
			results = append(results, memoryrails.SearchResult{
				Memory:     mem,
				Similarity: sim,
			})
		}
	}

	// Sort by similarity descending
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Similarity > results[i].Similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func (s *Store) List(ctx context.Context, opts memoryrails.ListOptions) ([]*memoryrails.Memory, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	query := "SELECT id, content, type, embedding, metadata, importance, access_count, last_accessed_at, created_at, updated_at FROM memories WHERE 1=1"
	var args []interface{}

	if opts.Type != "" {
		query += " AND type = ?"
		args = append(args, string(opts.Type))
	}

	orderBy := "created_at"
	if opts.OrderBy != "" {
		orderBy = opts.OrderBy
	}
	if opts.Descending {
		query += " ORDER BY " + orderBy + " DESC"
	} else {
		query += " ORDER BY " + orderBy
	}

	query += " LIMIT ?"
	args = append(args, limit)

	if opts.Offset > 0 {
		query += " OFFSET ?"
		args = append(args, opts.Offset)
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite store: list failed: %w", err)
	}
	defer rows.Close()

	var memories []*memoryrails.Memory
	for rows.Next() {
		mem, err := s.scanRows(rows)
		if err != nil {
			continue
		}
		memories = append(memories, mem)
	}

	return memories, nil
}

func (s *Store) Delete(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, "DELETE FROM memories WHERE id = ?", id)
	return err
}

func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) scanRow(row *sql.Row) (*memoryrails.Memory, error) {
	var (
		id, content, memType, createdStr, updatedStr string
		embStr, metaStr, lastAccessStr               sql.NullString
		importance                                   float64
		accessCount                                  int
	)

	err := row.Scan(&id, &content, &memType, &embStr, &metaStr, &importance, &accessCount, &lastAccessStr, &createdStr, &updatedStr)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}

	return buildMemory(id, content, memType, embStr, metaStr, lastAccessStr, importance, accessCount, createdStr, updatedStr), nil
}

func (s *Store) scanRows(rows *sql.Rows) (*memoryrails.Memory, error) {
	var (
		id, content, memType, createdStr, updatedStr string
		embStr, metaStr, lastAccessStr               sql.NullString
		importance                                   float64
		accessCount                                  int
	)

	err := rows.Scan(&id, &content, &memType, &embStr, &metaStr, &importance, &accessCount, &lastAccessStr, &createdStr, &updatedStr)
	if err != nil {
		return nil, err
	}

	return buildMemory(id, content, memType, embStr, metaStr, lastAccessStr, importance, accessCount, createdStr, updatedStr), nil
}

func buildMemory(id, content, memType string, embStr, metaStr, lastAccessStr sql.NullString, importance float64, accessCount int, createdStr, updatedStr string) *memoryrails.Memory {
	mem := &memoryrails.Memory{
		ID:          id,
		Content:     content,
		Type:        memoryrails.MemoryType(memType),
		Importance:  importance,
		AccessCount: accessCount,
	}

	if embStr.Valid {
		var emb []float32
		_ = json.Unmarshal([]byte(embStr.String), &emb)
		mem.Embedding = emb
	}

	if metaStr.Valid && metaStr.String != "" {
		var meta map[string]any
		_ = json.Unmarshal([]byte(metaStr.String), &meta)
		mem.Metadata = meta
	}

	if lastAccessStr.Valid {
		t, _ := time.Parse(time.RFC3339, lastAccessStr.String)
		mem.LastAccessedAt = t
	}

	mem.CreatedAt, _ = time.Parse(time.RFC3339, createdStr)
	mem.UpdatedAt, _ = time.Parse(time.RFC3339, updatedStr)

	return mem
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// formatEmbedding converts float32 slice to JSON string for storage.
func formatEmbedding(emb []float32) string {
	parts := make([]string, len(emb))
	for i, v := range emb {
		parts[i] = fmt.Sprintf("%f", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}
