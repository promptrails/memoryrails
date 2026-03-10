package memoryrails

import "context"

// SearchResult is a memory matched by similarity search.
type SearchResult struct {
	// Memory is the matched memory.
	Memory *Memory

	// Similarity is the cosine similarity score (0-1).
	Similarity float64
}

// SearchOptions configures a similarity search.
type SearchOptions struct {
	// Limit is the maximum number of results. Default: 10.
	Limit int

	// Threshold is the minimum similarity score (0-1). Default: 0.5.
	Threshold float64

	// Type filters results to a specific memory type. Empty = all types.
	Type MemoryType

	// Metadata filters results by metadata key-value pairs.
	Metadata map[string]any
}

// ListOptions configures a memory listing.
type ListOptions struct {
	// Limit is the maximum number of results. Default: 50.
	Limit int

	// Offset is the number of results to skip.
	Offset int

	// Type filters results to a specific memory type. Empty = all types.
	Type MemoryType

	// OrderBy is the field to sort by. Default: "created_at".
	OrderBy string

	// Descending reverses the sort order.
	Descending bool
}

// Store persists and retrieves memories with vector similarity search.
// Implementations must be safe for concurrent use.
type Store interface {
	// Put stores or updates a memory. If the memory has an ID that already
	// exists, it is updated. Otherwise, a new memory is created.
	Put(ctx context.Context, memory *Memory) error

	// Get retrieves a memory by ID. Returns nil if not found.
	Get(ctx context.Context, id string) (*Memory, error)

	// Search finds memories similar to the given embedding vector.
	Search(ctx context.Context, embedding []float32, opts SearchOptions) ([]SearchResult, error)

	// List returns memories matching the given options.
	List(ctx context.Context, opts ListOptions) ([]*Memory, error)

	// Delete removes a memory by ID.
	Delete(ctx context.Context, id string) error

	// Close releases any resources held by the store.
	Close() error
}
