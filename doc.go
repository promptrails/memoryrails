// Package memoryrails provides agent memory management for Go LLM applications.
//
// It stores, embeds, and retrieves memories using pluggable embedding providers
// and vector stores. Supports 5 memory types, importance scoring with decay,
// and semantic similarity search.
//
// # Quick Start
//
//	mgr := memoryrails.NewManager(embedder, store)
//
//	// Store a memory
//	mem, _ := mgr.Remember(ctx, "The user prefers dark mode.", memoryrails.TypeFact, nil)
//
//	// Recall relevant memories
//	results, _ := mgr.Recall(ctx, "What are the user's preferences?", memoryrails.RecallOptions{Limit: 5})
//	for _, r := range results {
//		fmt.Printf("%.2f: %s\n", r.Similarity, r.Memory.Content)
//	}
package memoryrails
