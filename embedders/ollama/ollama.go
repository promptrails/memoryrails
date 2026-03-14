package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	defaultBaseURL = "http://localhost:11434"
	defaultModel   = "nomic-embed-text"
)

// Embedder generates embeddings using Ollama's local API.
type Embedder struct {
	baseURL string
	model   string
	client  *http.Client
	dims    int
}

// Option configures the embedder.
type Option func(*Embedder)

// WithBaseURL sets a custom Ollama URL. Default: http://localhost:11434.
func WithBaseURL(url string) Option {
	return func(e *Embedder) { e.baseURL = url }
}

// WithModel sets the embedding model. Default: nomic-embed-text.
func WithModel(model string) Option {
	return func(e *Embedder) { e.model = model }
}

// WithDimensions sets the expected embedding dimensions.
// Default: 768 (nomic-embed-text).
func WithDimensions(dims int) Option {
	return func(e *Embedder) { e.dims = dims }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(e *Embedder) { e.client = client }
}

// New creates a new Ollama embedder.
func New(opts ...Option) *Embedder {
	e := &Embedder{
		baseURL: defaultBaseURL,
		model:   defaultModel,
		client:  &http.Client{Timeout: 30 * 1e9},
		dims:    768,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody, err := json.Marshal(embedRequest{
		Model: e.model,
		Input: text,
	})
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/embed", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ollama embedder: read error: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama embedder: API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result embedResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("ollama embedder: parse error: %w", err)
	}

	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama embedder: no embeddings returned")
	}

	return result.Embeddings[0], nil
}

func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := e.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("ollama embedder: batch item %d: %w", i, err)
		}
		results[i] = emb
	}
	return results, nil
}

func (e *Embedder) Dimensions() int { return e.dims }

type embedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}
