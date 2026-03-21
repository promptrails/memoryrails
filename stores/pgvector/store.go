package pgvector

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pgvector/pgvector-go"
	"gorm.io/gorm"

	"github.com/promptrails/memoryrails"
)

// MemoryRecord is the GORM model for the memories table.
type MemoryRecord struct {
	ID             string          `gorm:"primaryKey;size:64"`
	Content        string          `gorm:"type:text;not null"`
	Type           string          `gorm:"size:50;not null;index"`
	Embedding      pgvector.Vector `gorm:"type:vector"`
	Metadata       string          `gorm:"type:jsonb"` // JSON string
	Importance     float64         `gorm:"default:0.5"`
	AccessCount    int             `gorm:"default:0"`
	LastAccessedAt *time.Time
	CreatedAt      time.Time `gorm:"index"`
	UpdatedAt      time.Time
}

// TableName returns the table name.
func (MemoryRecord) TableName() string { return "memories" }

// Store implements memoryrails.Store using PostgreSQL with pgvector.
type Store struct {
	db   *gorm.DB
	dims int
}

// Option configures the store.
type Option func(*Store)

// WithDimensions sets the vector dimensions for the embedding column.
// Default: 1536.
func WithDimensions(dims int) Option {
	return func(s *Store) { s.dims = dims }
}

// New creates a new pgvector store. It auto-migrates the table and
// creates the pgvector extension and HNSW index if they don't exist.
func New(db *gorm.DB, opts ...Option) (*Store, error) {
	s := &Store{db: db, dims: 1536}
	for _, opt := range opts {
		opt(s)
	}

	// Enable pgvector extension
	if err := db.Exec("CREATE EXTENSION IF NOT EXISTS vector").Error; err != nil {
		return nil, fmt.Errorf("pgvector store: failed to create extension: %w", err)
	}

	// Auto-migrate
	if err := db.AutoMigrate(&MemoryRecord{}); err != nil {
		return nil, fmt.Errorf("pgvector store: migration failed: %w", err)
	}

	// Alter embedding column to correct dimensions
	alterSQL := fmt.Sprintf("ALTER TABLE memories ALTER COLUMN embedding TYPE vector(%d)", s.dims)
	_ = db.Exec(alterSQL).Error // ignore error if already correct

	// Create HNSW index
	_ = db.Exec("CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING hnsw (embedding vector_cosine_ops)").Error

	return s, nil
}

func (s *Store) Put(ctx context.Context, mem *memoryrails.Memory) error {
	metadataJSON := "{}"
	if mem.Metadata != nil {
		// Simple JSON encode
		parts := make([]string, 0, len(mem.Metadata))
		for k, v := range mem.Metadata {
			parts = append(parts, fmt.Sprintf(`"%s":"%v"`, k, v))
		}
		metadataJSON = "{" + strings.Join(parts, ",") + "}"
	}

	record := MemoryRecord{
		ID:             mem.ID,
		Content:        mem.Content,
		Type:           string(mem.Type),
		Embedding:      pgvector.NewVector(mem.Embedding),
		Metadata:       metadataJSON,
		Importance:     mem.Importance,
		AccessCount:    mem.AccessCount,
		LastAccessedAt: timePtr(mem.LastAccessedAt),
		CreatedAt:      mem.CreatedAt,
		UpdatedAt:      mem.UpdatedAt,
	}

	result := s.db.WithContext(ctx).Save(&record)
	return result.Error
}

func (s *Store) Get(ctx context.Context, id string) (*memoryrails.Memory, error) {
	var record MemoryRecord
	result := s.db.WithContext(ctx).Where("id = ?", id).First(&record)
	if result.Error != nil {
		if result.Error == gorm.ErrRecordNotFound {
			return nil, nil
		}
		return nil, result.Error
	}
	return recordToMemory(&record), nil
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

	vec := pgvector.NewVector(embedding)

	query := s.db.WithContext(ctx).
		Table("memories").
		Select("*, 1 - (embedding <=> ?) as similarity", vec).
		Where("1 - (embedding <=> ?) >= ?", vec, threshold)

	if opts.Type != "" {
		query = query.Where("type = ?", string(opts.Type))
	}

	query = query.Order("similarity DESC").Limit(limit)

	type resultRow struct {
		MemoryRecord
		Similarity float64
	}

	var rows []resultRow
	if err := query.Find(&rows).Error; err != nil {
		return nil, fmt.Errorf("pgvector store: search failed: %w", err)
	}

	results := make([]memoryrails.SearchResult, len(rows))
	for i, row := range rows {
		results[i] = memoryrails.SearchResult{
			Memory:     recordToMemory(&row.MemoryRecord),
			Similarity: row.Similarity,
		}
	}

	return results, nil
}

func (s *Store) List(ctx context.Context, opts memoryrails.ListOptions) ([]*memoryrails.Memory, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	query := s.db.WithContext(ctx)

	if opts.Type != "" {
		query = query.Where("type = ?", string(opts.Type))
	}

	orderBy := "created_at"
	if opts.OrderBy != "" {
		orderBy = opts.OrderBy
	}
	if opts.Descending {
		orderBy += " DESC"
	}
	query = query.Order(orderBy)

	if opts.Offset > 0 {
		query = query.Offset(opts.Offset)
	}

	var records []MemoryRecord
	if err := query.Limit(limit).Find(&records).Error; err != nil {
		return nil, err
	}

	memories := make([]*memoryrails.Memory, len(records))
	for i, r := range records {
		memories[i] = recordToMemory(&r)
	}
	return memories, nil
}

func (s *Store) Delete(ctx context.Context, id string) error {
	return s.db.WithContext(ctx).Where("id = ?", id).Delete(&MemoryRecord{}).Error
}

// Close is a no-op; the caller manages the gorm.DB lifecycle.
func (s *Store) Close() error { return nil }

func recordToMemory(r *MemoryRecord) *memoryrails.Memory {
	mem := &memoryrails.Memory{
		ID:          r.ID,
		Content:     r.Content,
		Type:        memoryrails.MemoryType(r.Type),
		Embedding:   r.Embedding.Slice(),
		Importance:  r.Importance,
		AccessCount: r.AccessCount,
		CreatedAt:   r.CreatedAt,
		UpdatedAt:   r.UpdatedAt,
	}
	if r.LastAccessedAt != nil {
		mem.LastAccessedAt = *r.LastAccessedAt
	}
	return mem
}

func timePtr(t time.Time) *time.Time {
	if t.IsZero() {
		return nil
	}
	return &t
}
