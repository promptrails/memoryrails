package sqlite

import (
	"context"
	"database/sql"
	"fmt"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/promptrails/memoryrails"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("failed to open sqlite: %v", err)
	}
	store, err := New(db)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	return store
}

func TestStore_PutAndGet(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	mem := &memoryrails.Memory{
		ID:        "test-1",
		Content:   "Hello world",
		Type:      memoryrails.TypeFact,
		Embedding: []float32{0.1, 0.2, 0.3},
		Metadata:  map[string]any{"key": "value"},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.Put(context.Background(), mem); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	got, err := s.Get(context.Background(), "test-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if got == nil {
		t.Fatal("expected memory, got nil")
	}
	if got.Content != "Hello world" {
		t.Errorf("expected 'Hello world', got %q", got.Content)
	}
	if len(got.Embedding) != 3 {
		t.Errorf("expected 3d embedding, got %d", len(got.Embedding))
	}
}

func TestStore_GetNotFound(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()

	got, err := s.Get(context.Background(), "nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Error("expected nil for nonexistent")
	}
}

func TestStore_Search(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	ctx := context.Background()

	now := time.Now()
	_ = s.Put(ctx, &memoryrails.Memory{ID: "1", Content: "similar", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})
	_ = s.Put(ctx, &memoryrails.Memory{ID: "2", Content: "different", Type: memoryrails.TypeFact, Embedding: []float32{0, 1, 0}, CreatedAt: now, UpdatedAt: now})

	results, err := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{Threshold: 0.5, Limit: 10})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Memory.ID != "1" {
		t.Errorf("expected ID '1', got %q", results[0].Memory.ID)
	}
}

func TestStore_Delete(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	ctx := context.Background()

	now := time.Now()
	_ = s.Put(ctx, &memoryrails.Memory{ID: "1", Content: "bye", Type: memoryrails.TypeFact, CreatedAt: now, UpdatedAt: now})
	_ = s.Delete(ctx, "1")

	got, _ := s.Get(ctx, "1")
	if got != nil {
		t.Error("expected nil after delete")
	}
}

func TestStore_List(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		now := time.Now().Add(time.Duration(i) * time.Minute)
		_ = s.Put(ctx, &memoryrails.Memory{
			ID: fmt.Sprintf("m%d", i), Content: "content", Type: memoryrails.TypeFact,
			CreatedAt: now, UpdatedAt: now,
		})
	}

	list, err := s.List(ctx, memoryrails.ListOptions{Limit: 3})
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(list) != 3 {
		t.Errorf("expected 3, got %d", len(list))
	}
}

func TestStore_Update(t *testing.T) {
	s := newTestStore(t)
	defer s.Close()
	ctx := context.Background()

	now := time.Now()
	_ = s.Put(ctx, &memoryrails.Memory{ID: "1", Content: "original", Type: memoryrails.TypeFact, CreatedAt: now, UpdatedAt: now})
	_ = s.Put(ctx, &memoryrails.Memory{ID: "1", Content: "updated", Type: memoryrails.TypeFact, CreatedAt: now, UpdatedAt: time.Now()})

	got, _ := s.Get(ctx, "1")
	if got.Content != "updated" {
		t.Errorf("expected 'updated', got %q", got.Content)
	}
}
