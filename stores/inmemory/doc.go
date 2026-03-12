// Package inmemory provides an in-memory vector store for memoryrails.
//
// It uses brute-force cosine similarity search and is suitable for
// development, testing, and small-scale applications (< 10K memories).
// Data is not persisted across restarts.
package inmemory
