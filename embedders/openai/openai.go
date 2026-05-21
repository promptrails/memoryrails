package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const defaultBaseURL = "https://api.openai.com/v1/embeddings"

// Model names.
const (
	ModelSmall = "text-embedding-3-small" // 1536 dimensions
	ModelLarge = "text-embedding-3-large" // 3072 dimensions
)

var modelDimensions = map[string]int{
	ModelSmall: 1536,
	ModelLarge: 3072,
}

// Embedder generates embeddings using OpenAI's API.
type Embedder struct {
	apiKey       string
	model        string
	baseURL      string
	client       *http.Client
	dims         int
	dimsOverride int
}

// Option configures the embedder.
type Option func(*Embedder)

// WithModel sets the embedding model. Default: text-embedding-3-small.
func WithModel(model string) Option {
	return func(e *Embedder) {
		e.model = model
		if d, ok := modelDimensions[model]; ok {
			e.dims = d
		}
	}
}

// WithBaseURL sets a custom API URL.
func WithBaseURL(url string) Option {
	return func(e *Embedder) { e.baseURL = url }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(e *Embedder) { e.client = client }
}

// WithDimensions truncates the returned vector to the given length. Only
// supported by text-embedding-3-* models; the value is sent to the API via
// the `dimensions` request field and the embedder's reported dimensions are
// updated to match. Passing 0 leaves the request unchanged (full model size).
func WithDimensions(dims int) Option {
	return func(e *Embedder) {
		e.dims = dims
		e.dimsOverride = dims
	}
}

// New creates a new OpenAI embedder.
func New(apiKey string, opts ...Option) *Embedder {
	e := &Embedder{
		apiKey:  apiKey,
		model:   ModelSmall,
		baseURL: defaultBaseURL,
		client:  &http.Client{Timeout: 30 * 1e9},
		dims:    1536,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	results, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody, err := json.Marshal(embeddingRequest{
		Input:      texts,
		Model:      e.model,
		Dimensions: e.dimsOverride,
	})
	if err != nil {
		return nil, fmt.Errorf("openai embedder: marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("openai embedder: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai embedder: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai embedder: read error: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai embedder: API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result embeddingResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("openai embedder: parse error: %w", err)
	}

	embeddings := make([][]float32, len(result.Data))
	for i, d := range result.Data {
		embeddings[i] = d.Embedding
	}
	return embeddings, nil
}

func (e *Embedder) Dimensions() int { return e.dims }

type embeddingRequest struct {
	Input      []string `json:"input"`
	Model      string   `json:"model"`
	Dimensions int      `json:"dimensions,omitempty"`
}

type embeddingResponse struct {
	Data []embeddingData `json:"data"`
}

type embeddingData struct {
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}
