//go:build integration

package qdrant

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/promptrails/memoryrails"
)

func testURL() string {
	if url := os.Getenv("TEST_QDRANT_URL"); url != "" {
		return url
	}
	return "http://localhost:6333"
}

func newTestStore(t *testing.T) *Store {
	t.Helper()
	store, err := New(testURL(), "memoryrails_test", 3)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	return store
}

func TestStore_PutAndGet(t *testing.T) {
	s := newTestStore(t)

	now := time.Now()
	mem := &memoryrails.Memory{
		ID:         "qd-1",
		Content:    "Hello from Qdrant",
		Type:       memoryrails.TypeFact,
		Embedding:  []float32{0.1, 0.2, 0.3},
		Importance: 0.7,
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	if err := s.Put(context.Background(), mem); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	// Qdrant may need a moment to index
	time.Sleep(100 * time.Millisecond)

	got, err := s.Get(context.Background(), "qd-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if got == nil {
		t.Fatal("expected memory, got nil")
	}
	if got.Content != "Hello from Qdrant" {
		t.Errorf("expected content, got %q", got.Content)
	}
}

func TestStore_Search(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	now := time.Now()

	_ = s.Put(ctx, &memoryrails.Memory{ID: "qs1", Content: "similar", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})
	_ = s.Put(ctx, &memoryrails.Memory{ID: "qs2", Content: "different", Type: memoryrails.TypeFact, Embedding: []float32{0, 1, 0}, CreatedAt: now, UpdatedAt: now})

	time.Sleep(200 * time.Millisecond) // wait for indexing

	results, err := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{Threshold: 0.5, Limit: 10})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least 1 result")
	}
	if results[0].Memory.ID != "qs1" {
		t.Errorf("expected qs1, got %q", results[0].Memory.ID)
	}
}

func TestStore_Delete(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	now := time.Now()

	_ = s.Put(ctx, &memoryrails.Memory{ID: "qd-del", Content: "bye", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})

	if err := s.Delete(ctx, "qd-del"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
}

func TestStore_List(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	now := time.Now()

	for i := 0; i < 3; i++ {
		_ = s.Put(ctx, &memoryrails.Memory{
			ID: string(rune('x' + i)), Content: "list item", Type: memoryrails.TypeFact,
			Embedding: []float32{float32(i), 0, 0}, CreatedAt: now, UpdatedAt: now,
		})
	}

	time.Sleep(200 * time.Millisecond)

	list, err := s.List(ctx, memoryrails.ListOptions{Limit: 10})
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(list) == 0 {
		t.Error("expected some results")
	}
}

func TestStore_Close(t *testing.T) {
	s := newTestStore(t)
	if err := s.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
}
