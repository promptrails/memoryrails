package opensearch

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/promptrails/memoryrails"
)

// newTestStore spins up an httptest server with the given handler and returns a
// store pointed at it. The handler must answer the index-creation PUT made by
// New.
func newTestStore(t *testing.T, handler http.HandlerFunc) *Store {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)

	s, err := New(server.URL, "memories", 3,
		WithRegion("us-east-1"),
		WithStaticCredentials("AKIDEXAMPLE", "secret", ""),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestNew_CreatesIndexAndSigns(t *testing.T) {
	var sawCreate bool
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && r.URL.Path == "/memories" {
			sawCreate = true
			if r.Header.Get("Authorization") == "" {
				t.Error("index creation not signed")
			}
		}
		w.WriteHeader(http.StatusOK)
	})
	_ = s
	if !sawCreate {
		t.Error("index was not created on New")
	}
}

func TestStore_Put(t *testing.T) {
	var putBody map[string]any
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && strings.Contains(r.URL.Path, "/_doc/") {
			_ = json.NewDecoder(r.Body).Decode(&putBody)
		}
		w.WriteHeader(http.StatusOK)
	})

	mem := &memoryrails.Memory{
		ID:        "mem-1",
		Content:   "hello",
		Type:      memoryrails.TypeFact,
		Embedding: []float32{0.1, 0.2, 0.3},
		Metadata:  map[string]any{"user": "alice"},
		CreatedAt: time.Unix(1700000000, 0),
	}
	if err := s.Put(context.Background(), mem); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if putBody["content"] != "hello" {
		t.Errorf("content = %v", putBody["content"])
	}
	if putBody["type"] != "fact" {
		t.Errorf("type = %v", putBody["type"])
	}
	if _, ok := putBody["embedding"]; !ok {
		t.Error("missing embedding in put body")
	}
}

func TestStore_GetNotFound(t *testing.T) {
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"found":false}`))
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	mem, err := s.Get(context.Background(), "missing")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem != nil {
		t.Errorf("expected nil for missing doc, got %+v", mem)
	}
}

func TestStore_Get(t *testing.T) {
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"_id":   "mem-1",
				"found": true,
				"_source": map[string]any{
					"content":    "hi",
					"type":       "fact",
					"importance": 0.8,
					"embedding":  []float32{0.1, 0.2, 0.3},
					"metadata":   map[string]any{"k": "v"},
				},
			})
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	mem, err := s.Get(context.Background(), "mem-1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if mem == nil || mem.ID != "mem-1" || mem.Content != "hi" || mem.Type != memoryrails.TypeFact {
		t.Fatalf("unexpected memory: %+v", mem)
	}
	if mem.Importance != 0.8 {
		t.Errorf("importance = %v", mem.Importance)
	}
	if len(mem.Embedding) != 3 {
		t.Errorf("embedding len = %d", len(mem.Embedding))
	}
	if mem.Metadata["k"] != "v" {
		t.Errorf("metadata = %v", mem.Metadata)
	}
}

func TestStore_Search(t *testing.T) {
	var searchBody map[string]any
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/_search") {
			_ = json.NewDecoder(r.Body).Decode(&searchBody)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"hits": map[string]any{
					"hits": []map[string]any{
						{"_id": "mem-1", "_score": 0.95, "_source": map[string]any{"content": "a", "type": "fact"}},
						{"_id": "mem-2", "_score": 0.80, "_source": map[string]any{"content": "b", "type": "fact"}},
					},
				},
			})
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	results, err := s.Search(context.Background(), []float32{0.1, 0.2, 0.3}, memoryrails.SearchOptions{
		Limit:     5,
		Threshold: 0.5,
		Type:      memoryrails.TypeFact,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Memory.ID != "mem-1" || results[0].Similarity != 0.95 {
		t.Errorf("result[0] = %+v", results[0])
	}
	// Type filter must be present in the query body.
	if !strings.Contains(mustJSON(searchBody), "\"fact\"") {
		t.Errorf("type filter missing from query: %s", mustJSON(searchBody))
	}
	if _, ok := searchBody["min_score"]; !ok {
		t.Error("threshold not applied as min_score")
	}
}

func TestStore_List(t *testing.T) {
	var listBody map[string]any
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/_search") {
			_ = json.NewDecoder(r.Body).Decode(&listBody)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"hits": map[string]any{
					"hits": []map[string]any{
						{"_id": "mem-1", "_source": map[string]any{"content": "a"}},
					},
				},
			})
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	mems, err := s.List(context.Background(), memoryrails.ListOptions{Limit: 10, Descending: true})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(mems) != 1 || mems[0].ID != "mem-1" {
		t.Fatalf("unexpected list: %+v", mems)
	}
	if _, ok := listBody["sort"]; !ok {
		t.Error("list query missing sort")
	}
}

func TestStore_Delete(t *testing.T) {
	var deleted bool
	s := newTestStore(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			deleted = true
		}
		w.WriteHeader(http.StatusOK)
	})

	if err := s.Delete(context.Background(), "mem-1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if !deleted {
		t.Error("delete request not issued")
	}
}

func TestStore_IndexAlreadyExistsIsIgnored(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && r.URL.Path == "/memories" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"type":"resource_already_exists_exception"}}`))
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	_, err := New(server.URL, "memories", 3,
		WithRegion("us-east-1"),
		WithStaticCredentials("k", "s", ""),
	)
	if err != nil {
		t.Fatalf("expected already-exists to be ignored, got %v", err)
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}
