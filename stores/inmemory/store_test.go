package inmemory

import (
	"context"
	"testing"
	"time"

	"github.com/promptrails/memoryrails"
)

func newTestMemory(id string, embedding []float32) *memoryrails.Memory {
	return &memoryrails.Memory{
		ID:        id,
		Content:   "test content " + id,
		Type:      memoryrails.TypeFact,
		Embedding: embedding,
		Metadata:  map[string]any{"key": "value"},
		CreatedAt: time.Now(),
	}
}

func TestStore_PutAndGet(t *testing.T) {
	s := New()
	ctx := context.Background()

	mem := newTestMemory("1", []float32{1, 0, 0})
	if err := s.Put(ctx, mem); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got, err := s.Get(ctx, "1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatal("expected memory, got nil")
	}
	if got.Content != "test content 1" {
		t.Errorf("expected content, got %q", got.Content)
	}
}

func TestStore_GetNotFound(t *testing.T) {
	s := New()
	got, err := s.Get(context.Background(), "nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Error("expected nil for nonexistent memory")
	}
}

func TestStore_Delete(t *testing.T) {
	s := New()
	ctx := context.Background()

	_ = s.Put(ctx, newTestMemory("1", nil))
	_ = s.Delete(ctx, "1")

	got, _ := s.Get(ctx, "1")
	if got != nil {
		t.Error("expected nil after delete")
	}
}

func TestStore_Search(t *testing.T) {
	s := New()
	ctx := context.Background()

	_ = s.Put(ctx, newTestMemory("1", []float32{1, 0, 0}))
	_ = s.Put(ctx, newTestMemory("2", []float32{0.9, 0.1, 0}))
	_ = s.Put(ctx, newTestMemory("3", []float32{0, 1, 0}))

	results, err := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{
		Limit:     2,
		Threshold: 0.5,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Memory.ID != "1" {
		t.Errorf("expected most similar first, got %q", results[0].Memory.ID)
	}
	if results[0].Similarity < results[1].Similarity {
		t.Error("expected descending similarity order")
	}
}

func TestStore_Search_TypeFilter(t *testing.T) {
	s := New()
	ctx := context.Background()

	m1 := newTestMemory("1", []float32{1, 0, 0})
	m1.Type = memoryrails.TypeFact
	m2 := newTestMemory("2", []float32{0.9, 0.1, 0})
	m2.Type = memoryrails.TypeConversation

	_ = s.Put(ctx, m1)
	_ = s.Put(ctx, m2)

	results, _ := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{
		Type:      memoryrails.TypeFact,
		Threshold: 0.1,
	})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Memory.ID != "1" {
		t.Errorf("expected memory 1, got %q", results[0].Memory.ID)
	}
}

func TestStore_Search_MetadataFilter(t *testing.T) {
	s := New()
	ctx := context.Background()

	m1 := newTestMemory("1", []float32{1, 0, 0})
	m1.Metadata = map[string]any{"env": "prod"}
	m2 := newTestMemory("2", []float32{0.9, 0.1, 0})
	m2.Metadata = map[string]any{"env": "dev"}

	_ = s.Put(ctx, m1)
	_ = s.Put(ctx, m2)

	results, _ := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{
		Threshold: 0.1,
		Metadata:  map[string]any{"env": "prod"},
	})
	if len(results) != 1 || results[0].Memory.ID != "1" {
		t.Error("expected only prod memory")
	}
}

func TestStore_List(t *testing.T) {
	s := New()
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		m := newTestMemory(string(rune('a'+i)), nil)
		m.CreatedAt = time.Now().Add(time.Duration(i) * time.Minute)
		_ = s.Put(ctx, m)
	}

	list, err := s.List(ctx, memoryrails.ListOptions{Limit: 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(list) != 3 {
		t.Errorf("expected 3, got %d", len(list))
	}
}

func TestStore_List_Descending(t *testing.T) {
	s := New()
	ctx := context.Background()

	m1 := newTestMemory("old", nil)
	m1.CreatedAt = time.Now().Add(-time.Hour)
	m2 := newTestMemory("new", nil)
	m2.CreatedAt = time.Now()

	_ = s.Put(ctx, m1)
	_ = s.Put(ctx, m2)

	list, _ := s.List(ctx, memoryrails.ListOptions{Descending: true})
	if len(list) < 2 {
		t.Fatal("expected at least 2")
	}
	if list[0].ID != "new" {
		t.Error("expected newest first in descending order")
	}
}

func TestStore_List_Offset(t *testing.T) {
	s := New()
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		m := newTestMemory(string(rune('a'+i)), nil)
		m.CreatedAt = time.Now().Add(time.Duration(i) * time.Minute)
		_ = s.Put(ctx, m)
	}

	list, _ := s.List(ctx, memoryrails.ListOptions{Offset: 10})
	if list != nil {
		t.Error("expected nil for offset beyond range")
	}
}

func TestStore_Len(t *testing.T) {
	s := New()
	if s.Len() != 0 {
		t.Error("expected 0")
	}
	_ = s.Put(context.Background(), newTestMemory("1", nil))
	if s.Len() != 1 {
		t.Error("expected 1")
	}
}

func TestStore_Close(t *testing.T) {
	s := New()
	if err := s.Close(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors
	sim := cosineSimilarity([]float32{1, 0, 0}, []float32{1, 0, 0})
	if sim < 0.99 {
		t.Errorf("expected ~1, got %f", sim)
	}

	// Orthogonal vectors
	sim = cosineSimilarity([]float32{1, 0, 0}, []float32{0, 1, 0})
	if sim > 0.01 {
		t.Errorf("expected ~0, got %f", sim)
	}

	// Different lengths
	sim = cosineSimilarity([]float32{1, 0}, []float32{1, 0, 0})
	if sim != 0 {
		t.Errorf("expected 0 for mismatched lengths, got %f", sim)
	}

	// Empty
	sim = cosineSimilarity(nil, nil)
	if sim != 0 {
		t.Errorf("expected 0 for empty, got %f", sim)
	}

	// Zero vector
	sim = cosineSimilarity([]float32{0, 0, 0}, []float32{1, 0, 0})
	if sim != 0 {
		t.Errorf("expected 0 for zero vector, got %f", sim)
	}
}

func TestStore_Search_NoEmbedding(t *testing.T) {
	s := New()
	ctx := context.Background()
	_ = s.Put(ctx, newTestMemory("1", nil)) // no embedding

	results, _ := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{Threshold: 0.1})
	if len(results) != 0 {
		t.Error("expected no results for memory without embedding")
	}
}
