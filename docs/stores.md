# Vector Stores

All stores implement the `Store` interface:

```go
type Store interface {
    Put(ctx context.Context, memory *Memory) error
    Get(ctx context.Context, id string) (*Memory, error)
    Search(ctx context.Context, embedding []float32, opts SearchOptions) ([]SearchResult, error)
    List(ctx context.Context, opts ListOptions) ([]*Memory, error)
    Delete(ctx context.Context, id string) error
    Close() error
}
```

## In-Memory

Brute-force cosine similarity. Good for development, testing, and small datasets.

```go
import "github.com/promptrails/memoryrails/stores/inmemory"

store := inmemory.New()
defer store.Close()
```

**Characteristics:**
- No persistence (data lost on restart)
- O(n) search (brute-force cosine similarity)
- Thread-safe
- Suitable for < 10K memories

## Search Options

```go
results, _ := store.Search(ctx, embedding, memoryrails.SearchOptions{
    Limit:     10,       // max results
    Threshold: 0.5,      // min similarity (0-1)
    Type:      memoryrails.TypeFact, // filter by type
    Metadata:  map[string]any{"user_id": "123"}, // filter by metadata
})
```

## List Options

```go
memories, _ := store.List(ctx, memoryrails.ListOptions{
    Limit:      50,
    Offset:     0,
    Type:       memoryrails.TypeConversation,
    Descending: true, // newest first
})
```

## Custom Store

Implement the `Store` interface to add your own backend (pgvector, Qdrant, etc.):

```go
type PgVectorStore struct {
    db *gorm.DB
}

func (s *PgVectorStore) Search(ctx context.Context, embedding []float32, opts memoryrails.SearchOptions) ([]memoryrails.SearchResult, error) {
    // Use pgvector's <=> operator for cosine distance
    // SELECT *, 1 - (embedding <=> $1::vector) as similarity
    // FROM memories WHERE similarity >= $2
    // ORDER BY similarity DESC LIMIT $3
}
```
