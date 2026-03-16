# Scoring & Decay

The `scoring` package provides importance scoring with time-based decay for memory retrieval ranking.

## Decay Scorer

Combines similarity with time-based importance decay and access frequency:

```
Score = similarity × (1 - decayWeight) + decayedImportance × decayWeight + accessBoost
```

Where:
- `decayedImportance = importance × 0.5^(age / halfLife)`
- `accessBoost = min(accessCount × factor, maxBoost)`

```go
import "github.com/promptrails/memoryrails/scoring"

scorer := scoring.NewDecayScorer()

mgr := memoryrails.NewManager(embedder, store,
    memoryrails.WithScorer(scorer),
)
```

## Configuration

```go
scorer := &scoring.DecayScorer{
    HalfLife:          7 * 24 * time.Hour, // importance halves every 7 days
    DecayWeight:       0.3,                // 30% importance, 70% similarity
    AccessBoostFactor: 0.01,               // +0.01 per access
    MaxAccessBoost:    0.2,                // capped at +0.2
}
```

## How It Works

**Without scorer** (default): results ranked purely by cosine similarity.

**With decay scorer**:
- Recent memories score higher than old ones (same importance)
- Frequently accessed memories get a boost
- High-importance memories persist longer before decay

## Example

A memory created 7 days ago with importance 0.8:
- Decayed importance: `0.8 × 0.5^1 = 0.4`
- With similarity 0.9: `0.9 × 0.7 + 0.4 × 0.3 = 0.63 + 0.12 = 0.75`

Same memory just created:
- Decayed importance: `0.8 × 0.5^0 = 0.8`
- With similarity 0.9: `0.9 × 0.7 + 0.8 × 0.3 = 0.63 + 0.24 = 0.87`

## Custom Scorer

Implement the `Scorer` interface:

```go
type Scorer interface {
    Score(result memoryrails.SearchResult, now time.Time) float64
}
```
