package memoryrails

import (
	"context"
	"testing"
	"time"
)

// fixedScorer returns a constant score, so we can assert the manager actually
// routes results through the custom scorer during Recall.
type fixedScorer struct{ score float64 }

func (s fixedScorer) Score(_ SearchResult, _ time.Time) float64 { return s.score }

func TestManager_Recall_WithScorer(t *testing.T) {
	embedder := &mockEmbedder{vector: []float32{1, 0, 0}, dim: 3}
	store := newMockStore()
	mgr := NewManager(embedder, store, WithScorer(fixedScorer{score: 0.42}))

	_, _ = mgr.Remember(context.Background(), "fact 1", TypeFact, nil)
	_, _ = mgr.Remember(context.Background(), "fact 2", TypeFact, nil)

	results, err := mgr.Recall(context.Background(), "query", RecallOptions{Limit: 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	// The mock store returns Similarity 0.9; the custom scorer must override it.
	for _, r := range results {
		if r.Similarity != 0.42 {
			t.Errorf("Similarity = %f, want 0.42 (custom scorer applied)", r.Similarity)
		}
	}
}
