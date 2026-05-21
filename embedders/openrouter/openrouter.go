package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const (
	defaultBaseURL = "https://openrouter.ai/api/v1/embeddings"
	defaultModel   = "openai/text-embedding-3-small"
)

// Embedder generates embeddings using OpenRouter's OpenAI-compatible API.
type Embedder struct {
	apiKey  string
	model   string
	baseURL string
	httpRef string
	xTitle  string
	client  *http.Client
	dims    int
}

// Option configures the embedder.
type Option func(*Embedder)

// WithModel sets the model. Default: openai/text-embedding-3-small.
func WithModel(model string) Option { return func(e *Embedder) { e.model = model } }

// WithDimensions sets the expected embedding dimensions. Default: 1536.
func WithDimensions(dims int) Option { return func(e *Embedder) { e.dims = dims } }

// WithBaseURL sets a custom API URL.
func WithBaseURL(url string) Option { return func(e *Embedder) { e.baseURL = url } }

// WithHTTPReferer sets the optional HTTP-Referer header that OpenRouter uses
// for attribution and rankings on openrouter.ai.
func WithHTTPReferer(url string) Option { return func(e *Embedder) { e.httpRef = url } }

// WithAppTitle sets the optional X-Title header used by OpenRouter for
// attribution on openrouter.ai.
func WithAppTitle(title string) Option { return func(e *Embedder) { e.xTitle = title } }

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option { return func(e *Embedder) { e.client = client } }

// New creates a new OpenRouter embedder.
func New(apiKey string, opts ...Option) *Embedder {
	e := &Embedder{
		apiKey:  apiKey,
		model:   defaultModel,
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
	reqBody, err := json.Marshal(map[string]interface{}{
		"input": texts,
		"model": e.model,
	})
	if err != nil {
		return nil, fmt.Errorf("openrouter embedder: marshal error: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("openrouter embedder: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)
	if e.httpRef != "" {
		req.Header.Set("HTTP-Referer", e.httpRef)
	}
	if e.xTitle != "" {
		req.Header.Set("X-Title", e.xTitle)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openrouter embedder: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openrouter embedder: API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("openrouter embedder: parse error: %w", err)
	}

	embeddings := make([][]float32, len(result.Data))
	for i, d := range result.Data {
		embeddings[i] = d.Embedding
	}
	return embeddings, nil
}

func (e *Embedder) Dimensions() int { return e.dims }
