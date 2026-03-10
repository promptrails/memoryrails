package memoryrails

import "context"

// Embedder generates vector embeddings from text.
// Implementations should be safe for concurrent use.
type Embedder interface {
	// Embed generates a vector embedding for a single text.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch generates embeddings for multiple texts.
	// Returns a slice of embeddings in the same order as the input.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the dimensionality of the embedding vectors.
	Dimensions() int
}
