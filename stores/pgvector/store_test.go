//go:build integration

package pgvector

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/promptrails/memoryrails"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

func newTestStore(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("TEST_POSTGRES_DSN")
	if dsn == "" {
		dsn = "host=localhost port=5432 user=postgres password=postgres dbname=memoryrails_test sslmode=disable"
	}

	db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	if err != nil {
		t.Fatalf("failed to connect: %v", err)
	}

	// Clean table
	db.Exec("DROP TABLE IF EXISTS memories")

	store, err := New(db, WithDimensions(3))
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}
	return store
}

func TestStore_PutAndGet(t *testing.T) {
	s := newTestStore(t)

	now := time.Now()
	mem := &memoryrails.Memory{
		ID:         "pgv-1",
		Content:    "Hello from pgvector",
		Type:       memoryrails.TypeFact,
		Embedding:  []float32{0.1, 0.2, 0.3},
		Metadata:   map[string]any{"env": "test"},
		Importance: 0.8,
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	if err := s.Put(context.Background(), mem); err != nil {
		t.Fatalf("put failed: %v", err)
	}

	got, err := s.Get(context.Background(), "pgv-1")
	if err != nil {
		t.Fatalf("get failed: %v", err)
	}
	if got == nil {
		t.Fatal("expected memory, got nil")
	}
	if got.Content != "Hello from pgvector" {
		t.Errorf("expected content, got %q", got.Content)
	}
}

func TestStore_GetNotFound(t *testing.T) {
	s := newTestStore(t)
	got, err := s.Get(context.Background(), "nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Error("expected nil")
	}
}

func TestStore_Search(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	now := time.Now()

	_ = s.Put(ctx, &memoryrails.Memory{ID: "s1", Content: "similar", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})
	_ = s.Put(ctx, &memoryrails.Memory{ID: "s2", Content: "different", Type: memoryrails.TypeFact, Embedding: []float32{0, 1, 0}, CreatedAt: now, UpdatedAt: now})

	results, err := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{Threshold: 0.5, Limit: 10})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Memory.ID != "s1" {
		t.Errorf("expected s1, got %q", results[0].Memory.ID)
	}
	if results[0].Similarity < 0.9 {
		t.Errorf("expected high similarity, got %f", results[0].Similarity)
	}
}

func TestStore_Search_TypeFilter(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	now := time.Now()

	_ = s.Put(ctx, &memoryrails.Memory{ID: "f1", Content: "fact", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})
	_ = s.Put(ctx, &memoryrails.Memory{ID: "c1", Content: "conv", Type: memoryrails.TypeConversation, Embedding: []float32{0.9, 0.1, 0}, CreatedAt: now, UpdatedAt: now})

	results, _ := s.Search(ctx, []float32{1, 0, 0}, memoryrails.SearchOptions{Threshold: 0.1, Type: memoryrails.TypeFact})
	if len(results) != 1 || results[0].Memory.ID != "f1" {
		t.Error("expected only fact type")
	}
}

func TestStore_Delete(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()
	now := time.Now()

	_ = s.Put(ctx, &memoryrails.Memory{ID: "d1", Content: "delete me", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})
	_ = s.Delete(ctx, "d1")

	got, _ := s.Get(ctx, "d1")
	if got != nil {
		t.Error("expected nil after delete")
	}
}

func TestStore_List(t *testing.T) {
	s := newTestStore(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		now := time.Now().Add(time.Duration(i) * time.Minute)
		_ = s.Put(ctx, &memoryrails.Memory{
			ID: string(rune('a' + i)), Content: "content", Type: memoryrails.TypeFact,
			Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now,
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
	ctx := context.Background()
	now := time.Now()

	_ = s.Put(ctx, &memoryrails.Memory{ID: "u1", Content: "original", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: now})
	_ = s.Put(ctx, &memoryrails.Memory{ID: "u1", Content: "updated", Type: memoryrails.TypeFact, Embedding: []float32{1, 0, 0}, CreatedAt: now, UpdatedAt: time.Now()})

	got, _ := s.Get(ctx, "u1")
	if got.Content != "updated" {
		t.Errorf("expected 'updated', got %q", got.Content)
	}
}
