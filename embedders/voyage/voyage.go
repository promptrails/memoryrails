package voyage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const defaultBaseURL = "https://api.voyageai.com/v1/embeddings"

// Embedder generates embeddings using Voyage AI's API.
type Embedder struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
	dims    int
}

// Option configures the embedder.
type Option func(*Embedder)

// WithModel sets the model. Default: voyage-3.
func WithModel(model string) Option { return func(e *Embedder) { e.model = model } }

// WithBaseURL sets a custom API URL.
func WithBaseURL(url string) Option { return func(e *Embedder) { e.baseURL = url } }

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option { return func(e *Embedder) { e.client = client } }

// New creates a new Voyage AI embedder.
func New(apiKey string, opts ...Option) *Embedder {
	e := &Embedder{
		apiKey:  apiKey,
		model:   "voyage-3",
		baseURL: defaultBaseURL,
		client:  &http.Client{Timeout: 30 * 1e9},
		dims:    1024,
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
	reqBody, _ := json.Marshal(map[string]interface{}{
		"input": texts,
		"model": e.model,
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("voyage embedder: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("voyage embedder: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("voyage embedder: API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("voyage embedder: parse error: %w", err)
	}

	embeddings := make([][]float32, len(result.Data))
	for i, d := range result.Data {
		embeddings[i] = d.Embedding
	}
	return embeddings, nil
}

func (e *Embedder) Dimensions() int { return e.dims }
