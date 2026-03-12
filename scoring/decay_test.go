package scoring

import (
	"testing"
	"time"

	"github.com/promptrails/memoryrails"
)

func TestDecayScorer_RecentMemoryHigherScore(t *testing.T) {
	scorer := NewDecayScorer()
	now := time.Now()

	recent := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance: 0.8,
			CreatedAt:  now.Add(-1 * time.Hour),
		},
		Similarity: 0.9,
	}

	old := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance: 0.8,
			CreatedAt:  now.Add(-30 * 24 * time.Hour), // 30 days ago
		},
		Similarity: 0.9,
	}

	recentScore := scorer.Score(recent, now)
	oldScore := scorer.Score(old, now)

	if recentScore <= oldScore {
		t.Errorf("expected recent (%.3f) > old (%.3f)", recentScore, oldScore)
	}
}

func TestDecayScorer_AccessBoost(t *testing.T) {
	scorer := NewDecayScorer()
	now := time.Now()

	noAccess := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance:  0.5,
			CreatedAt:   now,
			AccessCount: 0,
		},
		Similarity: 0.8,
	}

	manyAccess := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance:  0.5,
			CreatedAt:   now,
			AccessCount: 10,
		},
		Similarity: 0.8,
	}

	noScore := scorer.Score(noAccess, now)
	manyScore := scorer.Score(manyAccess, now)

	if manyScore <= noScore {
		t.Errorf("expected accessed (%.3f) > no access (%.3f)", manyScore, noScore)
	}
}

func TestDecayScorer_AccessBoostCapped(t *testing.T) {
	scorer := NewDecayScorer()
	now := time.Now()

	result := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance:  0.5,
			CreatedAt:   now,
			AccessCount: 1000, // huge access count
		},
		Similarity: 0.8,
	}

	score := scorer.Score(result, now)
	if score > 1.0 {
		t.Errorf("expected score <= 1.0, got %.3f", score)
	}
}

func TestDecayScorer_ZeroImportance(t *testing.T) {
	scorer := NewDecayScorer()
	now := time.Now()

	result := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance: 0,
			CreatedAt:  now,
		},
		Similarity: 0.8,
	}

	score := scorer.Score(result, now)
	if score <= 0 {
		t.Errorf("expected positive score even with zero importance, got %.3f", score)
	}
}

func TestDecayScorer_HalfLifeDecay(t *testing.T) {
	scorer := NewDecayScorer()
	scorer.HalfLife = 24 * time.Hour // 1 day half-life
	now := time.Now()

	result := memoryrails.SearchResult{
		Memory: &memoryrails.Memory{
			Importance: 1.0,
			CreatedAt:  now.Add(-24 * time.Hour), // exactly 1 half-life ago
		},
		Similarity: 0.0, // isolate decay effect
	}

	score := scorer.Score(result, now)
	// With similarity=0 and decay_weight=0.3: score ≈ 0 * 0.7 + 0.5 * 0.3 = 0.15
	if score < 0.1 || score > 0.2 {
		t.Errorf("expected score around 0.15 for half-decayed importance, got %.3f", score)
	}
}
