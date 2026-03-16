package inmemory

import (
	"context"
	"math"
	"sort"
	"sync"

	"github.com/promptrails/memoryrails"
)

// Store is an in-memory vector store using brute-force cosine similarity.
type Store struct {
	mu       sync.RWMutex
	memories map[string]*memoryrails.Memory
}

// New creates a new in-memory store.
func New() *Store {
	return &Store{
		memories: make(map[string]*memoryrails.Memory),
	}
}

func (s *Store) Put(_ context.Context, memory *memoryrails.Memory) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	// Deep copy to avoid external mutation
	m := *memory
	s.memories[m.ID] = &m
	return nil
}

func (s *Store) Get(_ context.Context, id string) (*memoryrails.Memory, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	m, ok := s.memories[id]
	if !ok {
		return nil, nil
	}
	cpy := *m
	return &cpy, nil
}

func (s *Store) Search(_ context.Context, embedding []float32, opts memoryrails.SearchOptions) ([]memoryrails.SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	var results []memoryrails.SearchResult

	for _, m := range s.memories {
		// Filter by type
		if opts.Type != "" && m.Type != opts.Type {
			continue
		}

		// Filter by metadata
		if !matchMetadata(m.Metadata, opts.Metadata) {
			continue
		}

		if len(m.Embedding) == 0 {
			continue
		}

		sim := cosineSimilarity(embedding, m.Embedding)
		if sim >= threshold {
			cpy := *m
			results = append(results, memoryrails.SearchResult{
				Memory:     &cpy,
				Similarity: sim,
			})
		}
	}

	// Sort by similarity descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func (s *Store) List(_ context.Context, opts memoryrails.ListOptions) ([]*memoryrails.Memory, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	var all []*memoryrails.Memory
	for _, m := range s.memories {
		if opts.Type != "" && m.Type != opts.Type {
			continue
		}
		cpy := *m
		all = append(all, &cpy)
	}

	// Sort by created_at
	sort.Slice(all, func(i, j int) bool {
		if opts.Descending {
			return all[i].CreatedAt.After(all[j].CreatedAt)
		}
		return all[i].CreatedAt.Before(all[j].CreatedAt)
	})

	// Apply offset
	if opts.Offset > 0 && opts.Offset < len(all) {
		all = all[opts.Offset:]
	} else if opts.Offset >= len(all) {
		return nil, nil
	}

	if len(all) > limit {
		all = all[:limit]
	}

	return all, nil
}

func (s *Store) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.memories, id)
	return nil
}

// Close is a no-op for the in-memory store.
func (s *Store) Close() error {
	return nil
}

// Len returns the number of memories in the store.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.memories)
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

func matchMetadata(memMeta, filterMeta map[string]any) bool {
	if len(filterMeta) == 0 {
		return true
	}
	if memMeta == nil {
		return false
	}
	for k, v := range filterMeta {
		if memMeta[k] != v {
			return false
		}
	}
	return true
}
