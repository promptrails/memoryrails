package memoryrails

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// RecallOptions configures a recall query.
type RecallOptions struct {
	// Limit is the maximum number of results. Default: 10.
	Limit int

	// Threshold is the minimum similarity score (0-1). Default: 0.5.
	Threshold float64

	// Type filters results to a specific memory type.
	Type MemoryType

	// Metadata filters results by metadata key-value pairs.
	Metadata map[string]any

	// UpdateAccess increments the access count and timestamp of returned memories.
	// Default: true.
	UpdateAccess *bool
}

// RememberOptions configures how a memory is stored.
type RememberOptions struct {
	// ID sets a custom ID. If empty, a random ID is generated.
	ID string

	// Importance sets the initial importance (0-1). Default: 0.5.
	Importance *float64
}

// Scorer computes a final retrieval score combining similarity and importance.
type Scorer interface {
	// Score computes the final score for a search result.
	Score(result SearchResult, now time.Time) float64
}

// Manager orchestrates embedding generation, storage, and retrieval.
type Manager struct {
	embedder Embedder
	store    Store
	scorer   Scorer
}

// ManagerOption configures the manager.
type ManagerOption func(*Manager)

// WithScorer sets a custom scorer for retrieval ranking.
func WithScorer(scorer Scorer) ManagerOption {
	return func(m *Manager) {
		m.scorer = scorer
	}
}

// NewManager creates a new memory manager.
func NewManager(embedder Embedder, store Store, opts ...ManagerOption) *Manager {
	m := &Manager{
		embedder: embedder,
		store:    store,
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// Remember stores a new memory. The content is embedded automatically.
func (m *Manager) Remember(ctx context.Context, content string, memType MemoryType, metadata map[string]any, opts ...RememberOptions) (*Memory, error) {
	embedding, err := m.embedder.Embed(ctx, content)
	if err != nil {
		return nil, fmt.Errorf("memoryrails: embedding failed: %w", err)
	}

	var opt RememberOptions
	if len(opts) > 0 {
		opt = opts[0]
	}

	id := opt.ID
	if id == "" {
		id = generateID()
	}

	importance := 0.5
	if opt.Importance != nil {
		importance = *opt.Importance
	}

	now := time.Now()
	mem := &Memory{
		ID:          id,
		Content:     content,
		Type:        memType,
		Embedding:   embedding,
		Metadata:    metadata,
		Importance:  importance,
		AccessCount: 0,
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	if err := m.store.Put(ctx, mem); err != nil {
		return nil, fmt.Errorf("memoryrails: store put failed: %w", err)
	}

	return mem, nil
}

// Recall searches for memories relevant to the query.
func (m *Manager) Recall(ctx context.Context, query string, opts RecallOptions) ([]SearchResult, error) {
	embedding, err := m.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("memoryrails: embedding failed: %w", err)
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	results, err := m.store.Search(ctx, embedding, SearchOptions{
		Limit:     limit,
		Threshold: threshold,
		Type:      opts.Type,
		Metadata:  opts.Metadata,
	})
	if err != nil {
		return nil, fmt.Errorf("memoryrails: search failed: %w", err)
	}

	// Apply scorer if set
	if m.scorer != nil {
		now := time.Now()
		for i := range results {
			results[i].Similarity = m.scorer.Score(results[i], now)
		}
	}

	// Update access tracking
	updateAccess := opts.UpdateAccess == nil || *opts.UpdateAccess
	if updateAccess {
		now := time.Now()
		for _, r := range results {
			r.Memory.AccessCount++
			r.Memory.LastAccessedAt = now
			_ = m.store.Put(ctx, r.Memory)
		}
	}

	return results, nil
}

// Forget deletes a memory by ID.
func (m *Manager) Forget(ctx context.Context, id string) error {
	return m.store.Delete(ctx, id)
}

// Get retrieves a memory by ID.
func (m *Manager) Get(ctx context.Context, id string) (*Memory, error) {
	return m.store.Get(ctx, id)
}

// List returns memories matching the given options.
func (m *Manager) List(ctx context.Context, opts ListOptions) ([]*Memory, error) {
	return m.store.List(ctx, opts)
}

func generateID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
