package memoryrails

import (
	"context"
	"testing"
)

type mockEmbedder struct {
	vector []float32
	dim    int
}

func (m *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return m.vector, nil
}

func (m *mockEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = m.vector
	}
	return result, nil
}

func (m *mockEmbedder) Dimensions() int { return m.dim }

type mockStore struct {
	memories map[string]*Memory
}

func newMockStore() *mockStore {
	return &mockStore{memories: make(map[string]*Memory)}
}

func (s *mockStore) Put(_ context.Context, mem *Memory) error {
	s.memories[mem.ID] = mem
	return nil
}

func (s *mockStore) Get(_ context.Context, id string) (*Memory, error) {
	return s.memories[id], nil
}

func (s *mockStore) Search(_ context.Context, _ []float32, _ SearchOptions) ([]SearchResult, error) {
	var results []SearchResult
	for _, m := range s.memories {
		results = append(results, SearchResult{Memory: m, Similarity: 0.9})
	}
	return results, nil
}

func (s *mockStore) List(_ context.Context, _ ListOptions) ([]*Memory, error) {
	var list []*Memory
	for _, m := range s.memories {
		list = append(list, m)
	}
	return list, nil
}

func (s *mockStore) Delete(_ context.Context, id string) error {
	delete(s.memories, id)
	return nil
}

func (s *mockStore) Close() error { return nil }

func TestManager_Remember(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store)

	mem, err := mgr.Remember(context.Background(), "The sky is blue", TypeFact, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mem.Content != "The sky is blue" {
		t.Errorf("expected content, got %q", mem.Content)
	}
	if mem.Type != TypeFact {
		t.Errorf("expected TypeFact, got %q", mem.Type)
	}
	if mem.Importance != 0.5 {
		t.Errorf("expected default importance 0.5, got %f", mem.Importance)
	}
	if len(mem.Embedding) != 3 {
		t.Errorf("expected 3D embedding, got %d", len(mem.Embedding))
	}
	if mem.ID == "" {
		t.Error("expected generated ID")
	}
}

func TestManager_Remember_CustomOptions(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store)

	imp := 0.9
	mem, err := mgr.Remember(context.Background(), "Important fact", TypeFact, nil, RememberOptions{
		ID:         "custom-id",
		Importance: &imp,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if mem.ID != "custom-id" {
		t.Errorf("expected custom-id, got %q", mem.ID)
	}
	if mem.Importance != 0.9 {
		t.Errorf("expected importance 0.9, got %f", mem.Importance)
	}
}

func TestManager_Recall(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store)

	_, _ = mgr.Remember(context.Background(), "fact 1", TypeFact, nil)
	_, _ = mgr.Remember(context.Background(), "fact 2", TypeFact, nil)

	results, err := mgr.Recall(context.Background(), "query", RecallOptions{Limit: 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	// Access count should be incremented
	for _, r := range results {
		if r.Memory.AccessCount != 1 {
			t.Errorf("expected access count 1, got %d", r.Memory.AccessCount)
		}
	}
}

func TestManager_Recall_NoUpdateAccess(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store)

	_, _ = mgr.Remember(context.Background(), "fact", TypeFact, nil)

	no := false
	results, _ := mgr.Recall(context.Background(), "query", RecallOptions{
		Limit:        10,
		UpdateAccess: &no,
	})
	for _, r := range results {
		if r.Memory.AccessCount != 0 {
			t.Error("expected access count 0 when UpdateAccess is false")
		}
	}
}

func TestManager_Forget(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store)

	mem, _ := mgr.Remember(context.Background(), "forget me", TypeFact, nil)
	err := mgr.Forget(context.Background(), mem.ID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got, _ := mgr.Get(context.Background(), mem.ID)
	if got != nil {
		t.Error("expected nil after forget")
	}
}

func TestManager_List(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store)

	_, _ = mgr.Remember(context.Background(), "mem1", TypeFact, nil)
	_, _ = mgr.Remember(context.Background(), "mem2", TypeConversation, nil)

	list, err := mgr.List(context.Background(), ListOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(list) != 2 {
		t.Errorf("expected 2 memories, got %d", len(list))
	}
}

func TestGenerateID(t *testing.T) {
	id1 := generateID()
	id2 := generateID()
	if id1 == "" || id2 == "" {
		t.Error("expected non-empty IDs")
	}
	if id1 == id2 {
		t.Error("expected unique IDs")
	}
	if len(id1) != 32 {
		t.Errorf("expected 32-char hex ID, got %d chars", len(id1))
	}
}
