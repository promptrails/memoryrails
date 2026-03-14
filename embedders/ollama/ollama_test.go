package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedder_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req embedRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "nomic-embed-text" {
			t.Errorf("expected model nomic-embed-text, got %q", req.Model)
		}
		if req.Input != "hello world" {
			t.Errorf("expected input 'hello world', got %q", req.Input)
		}

		resp := embedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := New(WithBaseURL(server.URL))
	emb, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(emb) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(emb))
	}
}

func TestEmbedder_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := embedResponse{
			Embeddings: [][]float32{{0.1, 0.2, 0.3}},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := New(WithBaseURL(server.URL))
	results, err := e.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestEmbedder_EmptyResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := embedResponse{Embeddings: nil}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := New(WithBaseURL(server.URL))
	_, err := e.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for empty response")
	}
}

func TestEmbedder_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("server error"))
	}))
	defer server.Close()

	e := New(WithBaseURL(server.URL))
	_, err := e.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedder_Dimensions(t *testing.T) {
	e := New()
	if e.Dimensions() != 768 {
		t.Errorf("expected 768, got %d", e.Dimensions())
	}

	e2 := New(WithDimensions(1536))
	if e2.Dimensions() != 1536 {
		t.Errorf("expected 1536, got %d", e2.Dimensions())
	}
}

func TestEmbedder_Options(t *testing.T) {
	e := New(
		WithBaseURL("http://custom:11434"),
		WithModel("mxbai-embed-large"),
		WithDimensions(1024),
		WithHTTPClient(&http.Client{}),
	)
	if e.baseURL != "http://custom:11434" {
		t.Error("expected custom URL")
	}
	if e.model != "mxbai-embed-large" {
		t.Error("expected custom model")
	}
	if e.dims != 1024 {
		t.Error("expected 1024 dims")
	}
}
