package opensearch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/promptrails/memoryrails"
	"github.com/promptrails/memoryrails/internal/awssig"
)

const (
	defaultRegion  = "us-east-1"
	defaultService = "aoss" // OpenSearch Serverless; use "es" for managed domains
	defaultEngine  = "nmslib"
	defaultSpace   = "cosinesimil"
)

// Store implements memoryrails.Store using Amazon OpenSearch (Serverless or
// managed domains) k-NN search over its REST API, signed with AWS SigV4.
type Store struct {
	baseURL string
	index   string
	region  string
	service string
	creds   awssig.Credentials
	client  *http.Client
	engine  string
	space   string
}

// Option configures the store.
type Option func(*Store)

// WithRegion sets the AWS region. Defaults to AWS_REGION / AWS_DEFAULT_REGION,
// then "us-east-1".
func WithRegion(region string) Option {
	return func(s *Store) { s.region = region }
}

// WithService sets the SigV4 service name. Default "aoss" (OpenSearch
// Serverless); use "es" for managed OpenSearch domains.
func WithService(service string) Option {
	return func(s *Store) { s.service = service }
}

// WithStaticCredentials sets explicit AWS credentials. Defaults are read from
// AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.
func WithStaticCredentials(accessKeyID, secretAccessKey, sessionToken string) Option {
	return func(s *Store) {
		s.creds = awssig.Credentials{
			AccessKeyID:     accessKeyID,
			SecretAccessKey: secretAccessKey,
			SessionToken:    sessionToken,
		}
	}
}

// WithKNNEngine sets the k-NN engine and space type used when creating the
// index. Defaults: engine "nmslib", space "cosinesimil".
func WithKNNEngine(engine, spaceType string) Option {
	return func(s *Store) {
		s.engine = engine
		s.space = spaceType
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(s *Store) { s.client = client }
}

// New creates an OpenSearch store. endpoint is the collection/domain endpoint
// (e.g. "https://abc123.us-east-1.aoss.amazonaws.com"), index is the index name
// and dims is the embedding dimensionality. The index is created if missing.
func New(endpoint, index string, dims int, opts ...Option) (*Store, error) {
	s := &Store{
		baseURL: strings.TrimRight(endpoint, "/"),
		index:   index,
		region:  firstNonEmpty(os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION")),
		service: defaultService,
		creds: awssig.Credentials{
			AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
			SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
			SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
		},
		client: &http.Client{Timeout: 30 * time.Second},
		engine: defaultEngine,
		space:  defaultSpace,
	}
	for _, opt := range opts {
		opt(s)
	}
	if s.region == "" {
		s.region = defaultRegion
	}

	if err := s.ensureIndex(dims); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Store) Put(ctx context.Context, mem *memoryrails.Memory) error {
	doc := map[string]any{
		"content":          mem.Content,
		"type":             string(mem.Type),
		"importance":       mem.Importance,
		"access_count":     mem.AccessCount,
		"last_accessed_at": mem.LastAccessedAt.Format(time.RFC3339),
		"created_at":       mem.CreatedAt.Format(time.RFC3339),
		"updated_at":       mem.UpdatedAt.Format(time.RFC3339),
		"embedding":        mem.Embedding,
	}
	if mem.Metadata != nil {
		doc["metadata"] = mem.Metadata
	}

	_, err := s.request(ctx, http.MethodPut,
		fmt.Sprintf("/%s/_doc/%s", s.index, url.PathEscape(mem.ID)), doc)
	return err
}

func (s *Store) Get(ctx context.Context, id string) (*memoryrails.Memory, error) {
	resp, err := s.request(ctx, http.MethodGet,
		fmt.Sprintf("/%s/_doc/%s", s.index, url.PathEscape(id)), nil)
	if err != nil {
		// A missing document returns 404; treat as not found rather than error.
		if isNotFound(err) {
			return nil, nil
		}
		return nil, err
	}

	var hit struct {
		ID     string         `json:"_id"`
		Found  bool           `json:"found"`
		Source map[string]any `json:"_source"`
	}
	if err := json.Unmarshal(resp, &hit); err != nil {
		return nil, fmt.Errorf("opensearch store: parse error: %w", err)
	}
	if !hit.Found {
		return nil, nil
	}
	return sourceToMemory(hit.ID, hit.Source), nil
}

func (s *Store) Search(ctx context.Context, embedding []float32, opts memoryrails.SearchOptions) ([]memoryrails.SearchResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 10
	}

	knn := map[string]any{
		"embedding": map[string]any{"vector": embedding, "k": limit},
	}
	query := map[string]any{
		"bool": map[string]any{
			"must": []map[string]any{{"knn": knn}},
		},
	}
	if filters := termFilters(opts.Type, opts.Metadata); len(filters) > 0 {
		query["bool"].(map[string]any)["filter"] = filters
	}

	body := map[string]any{
		"size":  limit,
		"query": query,
	}
	if opts.Threshold > 0 {
		body["min_score"] = opts.Threshold
	}

	resp, err := s.request(ctx, http.MethodPost, fmt.Sprintf("/%s/_search", s.index), body)
	if err != nil {
		return nil, err
	}

	hits, err := parseHits(resp)
	if err != nil {
		return nil, err
	}
	results := make([]memoryrails.SearchResult, len(hits))
	for i, h := range hits {
		results[i] = memoryrails.SearchResult{
			Memory:     sourceToMemory(h.ID, h.Source),
			Similarity: h.Score,
		}
	}
	return results, nil
}

func (s *Store) List(ctx context.Context, opts memoryrails.ListOptions) ([]*memoryrails.Memory, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	orderBy := opts.OrderBy
	if orderBy == "" {
		orderBy = "created_at"
	}
	order := "asc"
	if opts.Descending {
		order = "desc"
	}

	query := map[string]any{"match_all": map[string]any{}}
	if filters := termFilters(opts.Type, nil); len(filters) > 0 {
		query = map[string]any{"bool": map[string]any{"filter": filters}}
	}

	body := map[string]any{
		"size":  limit,
		"from":  opts.Offset,
		"query": query,
		"sort":  []map[string]any{{orderBy: map[string]any{"order": order}}},
	}

	resp, err := s.request(ctx, http.MethodPost, fmt.Sprintf("/%s/_search", s.index), body)
	if err != nil {
		return nil, err
	}

	hits, err := parseHits(resp)
	if err != nil {
		return nil, err
	}
	memories := make([]*memoryrails.Memory, len(hits))
	for i, h := range hits {
		memories[i] = sourceToMemory(h.ID, h.Source)
	}
	return memories, nil
}

func (s *Store) Delete(ctx context.Context, id string) error {
	_, err := s.request(ctx, http.MethodDelete,
		fmt.Sprintf("/%s/_doc/%s", s.index, url.PathEscape(id)), nil)
	if err != nil && isNotFound(err) {
		return nil
	}
	return err
}

// Close is a no-op for the REST-based client.
func (s *Store) Close() error { return nil }

func (s *Store) ensureIndex(dims int) error {
	body := map[string]any{
		"settings": map[string]any{"index": map[string]any{"knn": true}},
		"mappings": map[string]any{"properties": map[string]any{
			"embedding": map[string]any{
				"type":      "knn_vector",
				"dimension": dims,
				"method": map[string]any{
					"name":       "hnsw",
					"engine":     s.engine,
					"space_type": s.space,
				},
			},
			"content":          map[string]any{"type": "text"},
			"type":             map[string]any{"type": "keyword"},
			"importance":       map[string]any{"type": "float"},
			"access_count":     map[string]any{"type": "integer"},
			"created_at":       map[string]any{"type": "date"},
			"updated_at":       map[string]any{"type": "date"},
			"last_accessed_at": map[string]any{"type": "date"},
		}},
	}
	// Ignore "already exists" errors, mirroring the qdrant store.
	if _, err := s.request(context.Background(), http.MethodPut, "/"+s.index, body); err != nil {
		if isAlreadyExists(err) {
			return nil
		}
		return err
	}
	return nil
}

func (s *Store) request(ctx context.Context, method, path string, body any) ([]byte, error) {
	var payload []byte
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("opensearch store: marshal error: %w", err)
		}
		payload = data
	}

	req, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("opensearch store: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	signer := &awssig.Signer{Credentials: s.creds, Region: s.region, Service: s.service}
	signer.Sign(req, payload, time.Now().UTC())

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("opensearch store: request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		return nil, &apiError{status: resp.StatusCode, body: string(respBody)}
	}
	return respBody, nil
}

type hit struct {
	ID     string
	Score  float64
	Source map[string]any
}

func parseHits(raw []byte) ([]hit, error) {
	var result struct {
		Hits struct {
			Hits []struct {
				ID     string         `json:"_id"`
				Score  float64        `json:"_score"`
				Source map[string]any `json:"_source"`
			} `json:"hits"`
		} `json:"hits"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("opensearch store: parse error: %w", err)
	}
	hits := make([]hit, len(result.Hits.Hits))
	for i, h := range result.Hits.Hits {
		hits[i] = hit{ID: h.ID, Score: h.Score, Source: h.Source}
	}
	return hits, nil
}

func termFilters(memType memoryrails.MemoryType, metadata map[string]any) []map[string]any {
	var filters []map[string]any
	if memType != "" {
		filters = append(filters, map[string]any{"term": map[string]any{"type": string(memType)}})
	}
	for k, v := range metadata {
		filters = append(filters, map[string]any{"term": map[string]any{"metadata." + k: v}})
	}
	return filters
}

func sourceToMemory(id string, src map[string]any) *memoryrails.Memory {
	mem := &memoryrails.Memory{ID: id}
	if v, ok := src["content"].(string); ok {
		mem.Content = v
	}
	if v, ok := src["type"].(string); ok {
		mem.Type = memoryrails.MemoryType(v)
	}
	if v, ok := src["importance"].(float64); ok {
		mem.Importance = v
	}
	if v, ok := src["access_count"].(float64); ok {
		mem.AccessCount = int(v)
	}
	if v, ok := src["created_at"].(string); ok {
		mem.CreatedAt, _ = time.Parse(time.RFC3339, v)
	}
	if v, ok := src["updated_at"].(string); ok {
		mem.UpdatedAt, _ = time.Parse(time.RFC3339, v)
	}
	if v, ok := src["last_accessed_at"].(string); ok {
		mem.LastAccessedAt, _ = time.Parse(time.RFC3339, v)
	}
	if v, ok := src["metadata"].(map[string]any); ok {
		mem.Metadata = v
	}
	if v, ok := src["embedding"].([]any); ok {
		mem.Embedding = toFloat32Slice(v)
	}
	return mem
}

func toFloat32Slice(v []any) []float32 {
	out := make([]float32, 0, len(v))
	for _, e := range v {
		if f, ok := e.(float64); ok {
			out = append(out, float32(f))
		}
	}
	return out
}

// apiError carries the HTTP status and body of a failed OpenSearch response so
// callers can distinguish 404s and 400 "already exists" from real failures.
type apiError struct {
	status int
	body   string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("opensearch store: API error (status %d): %s", e.status, e.body)
}

func isNotFound(err error) bool {
	if ae, ok := err.(*apiError); ok {
		return ae.status == http.StatusNotFound
	}
	return false
}

func isAlreadyExists(err error) bool {
	if ae, ok := err.(*apiError); ok {
		return ae.status == http.StatusBadRequest && strings.Contains(ae.body, "resource_already_exists_exception")
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
