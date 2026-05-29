package bedrock

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

	"github.com/promptrails/memoryrails/internal/awssig"
)

const (
	defaultRegion = "us-east-1"
	service       = "bedrock"

	// ModelTitanV2 is Amazon Titan Text Embeddings V2 (1024 dims by default,
	// also supports 256 and 512 via WithDimensions).
	ModelTitanV2 = "amazon.titan-embed-text-v2:0"
	// ModelTitanV1 is Amazon Titan Text Embeddings V1 (1536 dims, fixed).
	ModelTitanV1 = "amazon.titan-embed-text-v1"
	// ModelCohereEnglish is Cohere Embed English v3 on Bedrock (1024 dims).
	ModelCohereEnglish = "cohere.embed-english-v3"
	// ModelCohereMultilingual is Cohere Embed Multilingual v3 on Bedrock (1024 dims).
	ModelCohereMultilingual = "cohere.embed-multilingual-v3"
)

var modelDimensions = map[string]int{
	ModelTitanV2:            1024,
	ModelTitanV1:            1536,
	ModelCohereEnglish:      1024,
	ModelCohereMultilingual: 1024,
}

// Embedder generates embeddings using Amazon Bedrock's InvokeModel API.
// It supports the Amazon Titan and Cohere embedding model families and signs
// requests with AWS Signature V4 using only the standard library.
type Embedder struct {
	region    string
	creds     awssig.Credentials
	model     string
	baseURL   string
	client    *http.Client
	dims      int
	dimsSet   bool   // true when WithDimensions was applied (Titan V2 only)
	inputType string // Cohere input_type, default "search_document"
}

// Option configures the embedder.
type Option func(*Embedder)

// WithRegion sets the AWS region. Defaults to AWS_REGION / AWS_DEFAULT_REGION,
// then "us-east-1".
func WithRegion(region string) Option {
	return func(e *Embedder) { e.region = region }
}

// WithStaticCredentials sets explicit AWS credentials. Defaults are read from
// AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.
func WithStaticCredentials(accessKeyID, secretAccessKey, sessionToken string) Option {
	return func(e *Embedder) {
		e.creds = awssig.Credentials{
			AccessKeyID:     accessKeyID,
			SecretAccessKey: secretAccessKey,
			SessionToken:    sessionToken,
		}
	}
}

// WithModel sets the embedding model ID. Default: Titan Text Embeddings V2.
func WithModel(model string) Option {
	return func(e *Embedder) {
		e.model = model
		if d, ok := modelDimensions[model]; ok && !e.dimsSet {
			e.dims = d
		}
	}
}

// WithDimensions requests a specific output dimensionality. Only Amazon Titan
// V2 supports this (256, 512 or 1024); it is ignored for other models.
func WithDimensions(dims int) Option {
	return func(e *Embedder) {
		e.dims = dims
		e.dimsSet = true
	}
}

// WithInputType sets the Cohere input_type ("search_document", "search_query",
// "classification", "clustering"). Default: "search_document". Ignored by Titan.
func WithInputType(inputType string) Option {
	return func(e *Embedder) { e.inputType = inputType }
}

// WithBaseURL overrides the Bedrock runtime endpoint (mainly for tests).
func WithBaseURL(rawURL string) Option {
	return func(e *Embedder) { e.baseURL = strings.TrimRight(rawURL, "/") }
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(e *Embedder) { e.client = client }
}

// New creates a Bedrock embedder. With no options it reads region and
// credentials from the standard AWS environment variables and uses Titan V2.
func New(opts ...Option) *Embedder {
	e := &Embedder{
		region: firstNonEmpty(os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION")),
		creds: awssig.Credentials{
			AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
			SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
			SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
		},
		model:     ModelTitanV2,
		client:    &http.Client{Timeout: 30 * time.Second},
		dims:      modelDimensions[ModelTitanV2],
		inputType: "search_document",
	}
	for _, opt := range opts {
		opt(e)
	}
	if e.region == "" {
		e.region = defaultRegion
	}
	if e.baseURL == "" {
		e.baseURL = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", e.region)
	}
	return e
}

// Embed generates an embedding for a single text.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	results, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// EmbedBatch generates embeddings for multiple texts. Cohere models embed the
// whole batch in one request; Titan models embed one text per request (the
// Titan API accepts a single input string).
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if e.isCohere() {
		return e.embedCohere(ctx, texts)
	}
	return e.embedTitan(ctx, texts)
}

// Dimensions returns the dimensionality of the embedding vectors.
func (e *Embedder) Dimensions() int { return e.dims }

func (e *Embedder) isCohere() bool { return strings.HasPrefix(e.model, "cohere.") }

func (e *Embedder) embedTitan(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, text := range texts {
		body := titanRequest{InputText: text}
		if e.dimsSet && e.model == ModelTitanV2 {
			body.Dimensions = e.dims
			body.Normalize = true
		}
		raw, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("bedrock embedder: marshal error: %w", err)
		}
		resp, err := e.invoke(ctx, raw)
		if err != nil {
			return nil, err
		}
		var tr titanResponse
		if err := json.Unmarshal(resp, &tr); err != nil {
			return nil, fmt.Errorf("bedrock embedder: parse error: %w", err)
		}
		out[i] = tr.Embedding
	}
	return out, nil
}

func (e *Embedder) embedCohere(ctx context.Context, texts []string) ([][]float32, error) {
	raw, err := json.Marshal(cohereRequest{Texts: texts, InputType: e.inputType, Truncate: "END"})
	if err != nil {
		return nil, fmt.Errorf("bedrock embedder: marshal error: %w", err)
	}
	resp, err := e.invoke(ctx, raw)
	if err != nil {
		return nil, err
	}
	var cr cohereResponse
	if err := json.Unmarshal(resp, &cr); err != nil {
		return nil, fmt.Errorf("bedrock embedder: parse error: %w", err)
	}
	return cr.Embeddings, nil
}

func (e *Embedder) invoke(ctx context.Context, body []byte) ([]byte, error) {
	endpoint := e.baseURL + "/model/" + url.PathEscape(e.model) + "/invoke"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("bedrock embedder: request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	signer := &awssig.Signer{Credentials: e.creds, Region: e.region, Service: service}
	signer.Sign(req, body, time.Now().UTC())

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock embedder: request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bedrock embedder: API error (status %d): %s", resp.StatusCode, string(raw))
	}
	return raw, nil
}

type titanRequest struct {
	InputText  string `json:"inputText"`
	Dimensions int    `json:"dimensions,omitempty"`
	Normalize  bool   `json:"normalize,omitempty"`
}

type titanResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

type cohereRequest struct {
	Texts     []string `json:"texts"`
	InputType string   `json:"input_type"`
	Truncate  string   `json:"truncate,omitempty"`
}

type cohereResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
