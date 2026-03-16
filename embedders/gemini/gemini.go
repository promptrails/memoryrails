package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta/models"

// Embedder generates embeddings using Google's Gemini API.
type Embedder struct {
	apiKey  string
	model   string
	baseURL string
	client  *http.Client
	dims    int
}

// Option configures the embedder.
type Option func(*Embedder)

// WithModel sets the model. Default: text-embedding-004.
func WithModel(model string) Option { return func(e *Embedder) { e.model = model } }

// WithBaseURL sets a custom API URL.
func WithBaseURL(url string) Option { return func(e *Embedder) { e.baseURL = url } }

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option { return func(e *Embedder) { e.client = client } }

// New creates a new Gemini embedder.
func New(apiKey string, opts ...Option) *Embedder {
	e := &Embedder{
		apiKey:  apiKey,
		model:   "text-embedding-004",
		baseURL: defaultBaseURL,
		client:  &http.Client{Timeout: 30 * 1e9},
		dims:    768,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	url := fmt.Sprintf("%s/%s:embedContent?key=%s", e.baseURL, e.model, e.apiKey)

	reqBody, _ := json.Marshal(map[string]interface{}{
		"content": map[string]interface{}{
			"parts": []map[string]string{{"text": text}},
		},
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("gemini embedder: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini embedder: request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini embedder: API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("gemini embedder: parse error: %w", err)
	}

	return result.Embedding.Values, nil
}

func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := e.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("gemini embedder: batch item %d: %w", i, err)
		}
		results[i] = emb
	}
	return results, nil
}

func (e *Embedder) Dimensions() int { return e.dims }
