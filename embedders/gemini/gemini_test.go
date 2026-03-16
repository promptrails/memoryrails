package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedder_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := map[string]interface{}{
			"embedding": map[string]interface{}{
				"values": []float32{0.1, 0.2, 0.3},
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

func TestEmbedder_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte("unauthorized"))
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
	if e.Dimensions() <= 0 {
		t.Error("expected positive dimensions")
	}
}

func TestEmbedder_Options(t *testing.T) {
	e := New("key",
		WithBaseURL("http://custom"),
		WithHTTPClient(&http.Client{}),
	)
	if e.baseURL != "http://custom" {
		t.Error("expected custom URL")
	}
}
