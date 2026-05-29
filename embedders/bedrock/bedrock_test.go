package bedrock

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testEmbedder(serverURL string, opts ...Option) *Embedder {
	base := []Option{
		WithRegion("us-east-1"),
		WithStaticCredentials("AKIDEXAMPLE", "secret", ""),
		WithBaseURL(serverURL),
	}
	return New(append(base, opts...)...)
}

func TestEmbedder_Titan(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, ModelTitanV2) || !strings.HasSuffix(r.URL.Path, "/invoke") {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if r.Header.Get("Authorization") == "" {
			t.Error("request not signed")
		}
		var req titanRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.InputText == "" {
			t.Error("missing inputText")
		}
		_ = json.NewEncoder(w).Encode(titanResponse{Embedding: []float32{0.1, 0.2, 0.3}})
	}))
	defer server.Close()

	e := testEmbedder(server.URL)
	vec, err := e.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 3 {
		t.Errorf("expected 3 dims, got %d", len(vec))
	}
	if e.Dimensions() != 1024 {
		t.Errorf("expected default 1024 dims, got %d", e.Dimensions())
	}
}

func TestEmbedder_TitanBatchLoops(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		_ = json.NewEncoder(w).Encode(titanResponse{Embedding: []float32{float32(calls)}})
	}))
	defer server.Close()

	e := testEmbedder(server.URL)
	out, err := e.EmbedBatch(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if calls != 3 {
		t.Errorf("expected 3 Titan calls, got %d", calls)
	}
	if len(out) != 3 {
		t.Errorf("expected 3 results, got %d", len(out))
	}
}

func TestEmbedder_TitanV2Dimensions(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req titanRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Dimensions != 512 {
			t.Errorf("expected dimensions 512 in request, got %d", req.Dimensions)
		}
		if !req.Normalize {
			t.Error("expected normalize=true")
		}
		_ = json.NewEncoder(w).Encode(titanResponse{Embedding: make([]float32, 512)})
	}))
	defer server.Close()

	e := testEmbedder(server.URL, WithModel(ModelTitanV2), WithDimensions(512))
	if e.Dimensions() != 512 {
		t.Errorf("expected 512 dims, got %d", e.Dimensions())
	}
	if _, err := e.Embed(context.Background(), "hi"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
}

func TestEmbedder_Cohere(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req cohereRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if len(req.Texts) != 2 {
			t.Errorf("expected 2 texts in one call, got %d", len(req.Texts))
		}
		if req.InputType != "search_document" {
			t.Errorf("input_type = %q", req.InputType)
		}
		_ = json.NewEncoder(w).Encode(cohereResponse{Embeddings: [][]float32{{0.1}, {0.2}}})
	}))
	defer server.Close()

	e := testEmbedder(server.URL, WithModel(ModelCohereEnglish))
	out, err := e.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(out) != 2 {
		t.Errorf("expected 2 results, got %d", len(out))
	}
	if e.Dimensions() != 1024 {
		t.Errorf("expected 1024 dims, got %d", e.Dimensions())
	}
}

func TestEmbedder_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"message":"denied"}`))
	}))
	defer server.Close()

	e := testEmbedder(server.URL)
	_, err := e.Embed(context.Background(), "hi")
	if err == nil || !strings.Contains(err.Error(), "403") {
		t.Errorf("expected 403 error, got %v", err)
	}
}
