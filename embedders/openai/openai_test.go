package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedder_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected auth header")
		}

		var req embeddingRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != ModelSmall {
			t.Errorf("expected model %q, got %q", ModelSmall, req.Model)
		}
		if len(req.Input) != 1 {
			t.Errorf("expected 1 input, got %d", len(req.Input))
		}

		resp := embeddingResponse{
			Data: []embeddingData{
				{Embedding: []float32{0.1, 0.2, 0.3}, Index: 0},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := New("test-key", WithBaseURL(server.URL))
	emb, err := e.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(emb) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(emb))
	}
}

func TestEmbedder_EmbedBatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := embeddingResponse{
			Data: []embeddingData{
				{Embedding: []float32{0.1, 0.2, 0.3}, Index: 0},
				{Embedding: []float32{0.4, 0.5, 0.6}, Index: 1},
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e := New("key", WithBaseURL(server.URL))
	results, err := e.EmbedBatch(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
}

func TestEmbedder_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"invalid key"}`))
	}))
	defer server.Close()

	e := New("bad-key", WithBaseURL(server.URL))
	_, err := e.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestEmbedder_Dimensions(t *testing.T) {
	e := New("key")
	if e.Dimensions() != 1536 {
		t.Errorf("expected 1536, got %d", e.Dimensions())
	}

	e2 := New("key", WithModel(ModelLarge))
	if e2.Dimensions() != 3072 {
		t.Errorf("expected 3072, got %d", e2.Dimensions())
	}
}

func TestEmbedder_Options(t *testing.T) {
	e := New("key",
		WithBaseURL("http://custom"),
		WithModel(ModelLarge),
		WithHTTPClient(&http.Client{}),
	)
	if e.baseURL != "http://custom" {
		t.Error("expected custom URL")
	}
	if e.model != ModelLarge {
		t.Error("expected large model")
	}
}
