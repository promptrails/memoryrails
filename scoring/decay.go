package scoring

import (
	"math"
	"time"

	"github.com/promptrails/memoryrails"
)

// DecayScorer combines similarity with time-based importance decay
// and access frequency. It balances relevance (similarity) with
// recency (decay) and usage (access count).
//
// Final score = similarity * (1 - decayWeight) + decayedImportance * decayWeight + accessBoost
type DecayScorer struct {
	// HalfLife is the duration after which importance decays to 50%.
	// Default: 7 days.
	HalfLife time.Duration

	// DecayWeight controls how much importance affects the final score.
	// Range: 0-1. Default: 0.3.
	DecayWeight float64

	// AccessBoostFactor is the boost per access. Default: 0.01.
	AccessBoostFactor float64

	// MaxAccessBoost caps the total access boost. Default: 0.2.
	MaxAccessBoost float64
}

// NewDecayScorer creates a scorer with sensible defaults.
func NewDecayScorer() *DecayScorer {
	return &DecayScorer{
		HalfLife:          7 * 24 * time.Hour,
		DecayWeight:       0.3,
		AccessBoostFactor: 0.01,
		MaxAccessBoost:    0.2,
	}
}

// Score computes the final retrieval score.
func (s *DecayScorer) Score(result memoryrails.SearchResult, now time.Time) float64 {
	mem := result.Memory
	similarity := result.Similarity

	// Time-based exponential decay
	age := now.Sub(mem.CreatedAt)
	halfLife := s.HalfLife
	if halfLife <= 0 {
		halfLife = 7 * 24 * time.Hour
	}
	decayFactor := math.Pow(0.5, float64(age)/float64(halfLife))
	decayedImportance := mem.Importance * decayFactor

	// Access frequency boost
	accessBoost := float64(mem.AccessCount) * s.AccessBoostFactor
	if accessBoost > s.MaxAccessBoost {
		accessBoost = s.MaxAccessBoost
	}

	// Combine
	decayWeight := s.DecayWeight
	if decayWeight <= 0 {
		decayWeight = 0.3
	}

	score := similarity*(1-decayWeight) + decayedImportance*decayWeight + accessBoost

	// Clamp to [0, 1]
	if score > 1 {
		score = 1
	}
	if score < 0 {
		score = 0
	}

	return score
}
