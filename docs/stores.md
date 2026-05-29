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

## PostgreSQL + pgvector

Production-grade vector store using PostgreSQL with the pgvector extension. Uses GORM for database operations and HNSW indexing for fast approximate nearest neighbor search.

```go
import (
    "github.com/promptrails/memoryrails/stores/pgvector"
    "gorm.io/driver/postgres"
    "gorm.io/gorm"
)

db, _ := gorm.Open(postgres.Open("host=localhost user=postgres dbname=myapp sslmode=disable"), &gorm.Config{})

store, _ := pgvector.New(db, pgvector.WithDimensions(1536))
defer store.Close()
```

**Characteristics:**
- Persistent storage
- HNSW indexing with cosine similarity (`<=>` operator)
- Auto-creates pgvector extension and table
- Type filtering via SQL WHERE clause
- Suitable for millions of memories

**Requirements:**
- PostgreSQL 15+ with [pgvector extension](https://github.com/pgvector/pgvector)
- `CREATE EXTENSION vector` (auto-created by the store)

**Configuration:**

```go
// Custom vector dimensions (default: 1536 for OpenAI)
store, _ := pgvector.New(db, pgvector.WithDimensions(768)) // for Ollama/Gemini
```

## SQLite

Lightweight store using SQLite with in-process cosine similarity computation. No extensions required — embeddings are stored as JSON arrays and similarity is computed in Go.

```go
import (
    "database/sql"
    "github.com/promptrails/memoryrails/stores/sqlite"
    _ "github.com/mattn/go-sqlite3"
)

db, _ := sql.Open("sqlite3", "./memories.db")
store, _ := sqlite.New(db)
defer store.Close()
```

**Characteristics:**
- File-based persistence (or `:memory:` for in-process)
- O(n) search (brute-force, computed in Go)
- No extensions required
- Suitable for edge, CLI tools, single-machine apps (< 100K memories)

**In-memory SQLite (for testing):**

```go
db, _ := sql.Open("sqlite3", ":memory:")
store, _ := sqlite.New(db)
```

## Qdrant

High-performance vector database via Qdrant's REST API. Best for large-scale deployments with millions of vectors.

```go
import "github.com/promptrails/memoryrails/stores/qdrant"

store, _ := qdrant.New("http://localhost:6333", "memories", 1536)
defer store.Close()

// With authentication (Qdrant Cloud)
store, _ := qdrant.New("https://xyz.eu-west-1.aws.cloud.qdrant.io:6333", "memories", 1536,
    qdrant.WithAPIKey("your-api-key"),
)
```

**Characteristics:**
- Auto-creates collection with cosine distance
- Payload-based filtering (type, metadata)
- Scroll-based listing
- Suitable for millions+ memories

**Requirements:**
- Qdrant server ([Docker](https://qdrant.tech/documentation/quick-start/), Cloud, or self-hosted)

```bash
# Quick start with Docker
docker run -p 6333:6333 qdrant/qdrant
```

## Amazon OpenSearch

AWS-managed k-NN vector search over the OpenSearch REST API. Works with both
OpenSearch Serverless (`aoss`, the default) and managed domains (`es`).
Authenticates with AWS Signature V4 (no `aws-sdk-go` dependency); region and
credentials default to the standard AWS environment variables.

```go
import "github.com/promptrails/memoryrails/stores/opensearch"

// OpenSearch Serverless collection (service "aoss")
store, _ := opensearch.New(
    "https://abc123.us-east-1.aoss.amazonaws.com",
    "memories",
    1024, // embedding dimensions
    opensearch.WithRegion("us-east-1"),
)
defer store.Close()

// Managed OpenSearch domain
store, _ := opensearch.New(endpoint, "memories", 1024,
    opensearch.WithRegion("us-east-1"),
    opensearch.WithService("es"),
)
```

**Characteristics:**
- Auto-creates the index with a `knn_vector` mapping (engine/space configurable via `WithKNNEngine`)
- Term-based filtering on type and metadata
- `from`/`size` listing with sort
- Search `Similarity` is OpenSearch's normalized `_score`, not raw cosine

**Requirements:**
- An OpenSearch Serverless collection or managed domain, plus IAM credentials with data-access permissions

## Search Options

```go
results, _ := store.Search(ctx, embedding, memoryrails.SearchOptions{
    Limit:     10,                              // max results
    Threshold: 0.5,                             // min similarity (0-1)
    Type:      memoryrails.TypeFact,            // filter by type
    Metadata:  map[string]any{"user_id": "123"}, // filter by metadata (inmemory only)
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

## Choosing a Store

| Store | Persistence | Scale | Search | Dependencies |
|-------|------------|-------|--------|--------------|
| In-Memory | No | < 10K | Brute-force | None |
| SQLite | File | < 100K | Brute-force | go-sqlite3 |
| pgvector | PostgreSQL | Millions | HNSW (fast) | PostgreSQL + pgvector |
| Qdrant | Server | Millions+ | HNSW (fast) | Qdrant server |

## Custom Store

Implement the `Store` interface to add your own backend:

```go
type MyStore struct { /* ... */ }

func (s *MyStore) Put(ctx context.Context, memory *memoryrails.Memory) error { /* ... */ }
func (s *MyStore) Get(ctx context.Context, id string) (*memoryrails.Memory, error) { /* ... */ }
func (s *MyStore) Search(ctx context.Context, embedding []float32, opts memoryrails.SearchOptions) ([]memoryrails.SearchResult, error) { /* ... */ }
func (s *MyStore) List(ctx context.Context, opts memoryrails.ListOptions) ([]*memoryrails.Memory, error) { /* ... */ }
func (s *MyStore) Delete(ctx context.Context, id string) error { /* ... */ }
func (s *MyStore) Close() error { /* ... */ }
```
