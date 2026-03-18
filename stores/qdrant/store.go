package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/promptrails/memoryrails"
)

// Store implements memoryrails.Store using Qdrant's REST API.
type Store struct {
	baseURL    string
	collection string
	client     *http.Client
	apiKey     string
}

// Option configures the store.
type Option func(*Store)

// WithAPIKey sets authentication for Qdrant Cloud.
func WithAPIKey(key string) Option {
	return func(s *Store) { s.apiKey = key }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) { s.client = client }
}

// New creates a new Qdrant store.
// baseURL is the Qdrant server (e.g., "http://localhost:6333").
// collection is the collection name to use.
func New(baseURL, collection string, dims int, opts ...Option) (*Store, error) {
	s := &Store{
		baseURL:    baseURL,
		collection: collection,
		client:     &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(s)
	}

	// Create collection if not exists
	if err := s.ensureCollection(dims); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *Store) Put(ctx context.Context, mem *memoryrails.Memory) error {
	payload := map[string]interface{}{
		"content":          mem.Content,
		"type":             string(mem.Type),
		"importance":       mem.Importance,
		"access_count":     mem.AccessCount,
		"last_accessed_at": mem.LastAccessedAt.Format(time.RFC3339),
		"created_at":       mem.CreatedAt.Format(time.RFC3339),
		"updated_at":       mem.UpdatedAt.Format(time.RFC3339),
	}
	if mem.Metadata != nil {
		for k, v := range mem.Metadata {
			payload["meta_"+k] = v
		}
	}

	body := map[string]interface{}{
		"points": []map[string]interface{}{
			{
				"id":      mem.ID,
				"vector":  mem.Embedding,
				"payload": payload,
			},
		},
	}

	_, err := s.request(ctx, http.MethodPut,
		fmt.Sprintf("/collections/%s/points", s.collection), body)
	return err
}

func (s *Store) Get(ctx context.Context, id string) (*memoryrails.Memory, error) {
	resp, err := s.request(ctx, http.MethodGet,
		fmt.Sprintf("/collections/%s/points/%s", s.collection, id), nil)
	if err != nil {
		return nil, err
	}

	var result struct {
		Result struct {
			ID      string                 `json:"id"`
			Vector  []float32              `json:"vector"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("qdrant store: parse error: %w", err)
	}

	return payloadToMemory(result.Result.ID, result.Result.Vector, result.Result.Payload), nil
}

func (s *Store) Search(ctx context.Context, embedding []float32, opts memoryrails.SearchOptions) ([]memoryrails.SearchResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = 0.5
	}

	body := map[string]interface{}{
		"vector":          embedding,
		"limit":           limit,
		"score_threshold": threshold,
		"with_payload":    true,
		"with_vector":     true,
	}

	if opts.Type != "" {
		body["filter"] = map[string]interface{}{
			"must": []map[string]interface{}{
				{"key": "type", "match": map[string]string{"value": string(opts.Type)}},
			},
		}
	}

	resp, err := s.request(ctx, http.MethodPost,
		fmt.Sprintf("/collections/%s/points/search", s.collection), body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Result []struct {
			ID      string                 `json:"id"`
			Score   float64                `json:"score"`
			Vector  []float32              `json:"vector"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("qdrant store: parse error: %w", err)
	}

	results := make([]memoryrails.SearchResult, len(result.Result))
	for i, r := range result.Result {
		results[i] = memoryrails.SearchResult{
			Memory:     payloadToMemory(r.ID, r.Vector, r.Payload),
			Similarity: r.Score,
		}
	}

	return results, nil
}

func (s *Store) List(ctx context.Context, opts memoryrails.ListOptions) ([]*memoryrails.Memory, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	body := map[string]interface{}{
		"limit":        limit,
		"offset":       opts.Offset,
		"with_payload": true,
		"with_vector":  true,
	}

	if opts.Type != "" {
		body["filter"] = map[string]interface{}{
			"must": []map[string]interface{}{
				{"key": "type", "match": map[string]string{"value": string(opts.Type)}},
			},
		}
	}

	resp, err := s.request(ctx, http.MethodPost,
		fmt.Sprintf("/collections/%s/points/scroll", s.collection), body)
	if err != nil {
		return nil, err
	}

	var result struct {
		Result struct {
			Points []struct {
				ID      string                 `json:"id"`
				Vector  []float32              `json:"vector"`
				Payload map[string]interface{} `json:"payload"`
			} `json:"points"`
		} `json:"result"`
	}
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, fmt.Errorf("qdrant store: parse error: %w", err)
	}

	memories := make([]*memoryrails.Memory, len(result.Result.Points))
	for i, p := range result.Result.Points {
		memories[i] = payloadToMemory(p.ID, p.Vector, p.Payload)
	}

	return memories, nil
}

func (s *Store) Delete(ctx context.Context, id string) error {
	body := map[string]interface{}{
		"points": []string{id},
	}
	_, err := s.request(ctx, http.MethodPost,
		fmt.Sprintf("/collections/%s/points/delete", s.collection), body)
	return err
}

// Close is a no-op for the REST-based client.
func (s *Store) Close() error { return nil }

func (s *Store) ensureCollection(dims int) error {
	body := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     dims,
			"distance": "Cosine",
		},
	}
	// Ignore error if collection already exists
	_, _ = s.request(context.Background(), http.MethodPut,
		fmt.Sprintf("/collections/%s", s.collection), body)
	return nil
}

func (s *Store) request(ctx context.Context, method, path string, body interface{}) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("qdrant store: marshal error: %w", err)
		}
		reqBody = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, reqBody)
	if err != nil {
		return nil, fmt.Errorf("qdrant store: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("api-key", s.apiKey)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("qdrant store: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("qdrant store: API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

func payloadToMemory(id string, vector []float32, payload map[string]interface{}) *memoryrails.Memory {
	mem := &memoryrails.Memory{
		ID:        id,
		Embedding: vector,
	}

	if v, ok := payload["content"].(string); ok {
		mem.Content = v
	}
	if v, ok := payload["type"].(string); ok {
		mem.Type = memoryrails.MemoryType(v)
	}
	if v, ok := payload["importance"].(float64); ok {
		mem.Importance = v
	}
	if v, ok := payload["access_count"].(float64); ok {
		mem.AccessCount = int(v)
	}
	if v, ok := payload["created_at"].(string); ok {
		mem.CreatedAt, _ = time.Parse(time.RFC3339, v)
	}
	if v, ok := payload["updated_at"].(string); ok {
		mem.UpdatedAt, _ = time.Parse(time.RFC3339, v)
	}
	if v, ok := payload["last_accessed_at"].(string); ok {
		mem.LastAccessedAt, _ = time.Parse(time.RFC3339, v)
	}

	return mem
}
