# MemoryRails

Agent memory management for Go LLM applications. Pluggable embeddings + vector stores.

[![Go Reference](https://pkg.go.dev/badge/github.com/promptrails/memoryrails.svg)](https://pkg.go.dev/github.com/promptrails/memoryrails)
[![CI](https://github.com/promptrails/memoryrails/actions/workflows/ci.yml/badge.svg)](https://github.com/promptrails/memoryrails/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/promptrails/memoryrails)](https://goreportcard.com/report/github.com/promptrails/memoryrails)

```go
mgr := memoryrails.NewManager(embedder, store)

// Store a memory
mgr.Remember(ctx, "User prefers dark mode", memoryrails.TypeFact, nil)

// Recall relevant memories
results, _ := mgr.Recall(ctx, "What are the user's preferences?", memoryrails.RecallOptions{Limit: 5})
```

## Install

```bash
go get github.com/promptrails/memoryrails
```

## Features

- **5 memory types** — conversation, fact, procedure, episodic, semantic
- **5 embedding providers** — OpenAI, Ollama, Cohere, Gemini, Voyage AI
- **Pluggable vector stores** — in-memory (included), pgvector, SQLite, Qdrant
- **Importance scoring** — time-based decay + access frequency boost
- **Semantic search** — cosine similarity with configurable threshold
- **Access tracking** — automatic retrieval count and timestamp
- **Framework independent** — works with any Go LLM library

## Embedding Providers

| Provider | Package | Models |
|----------|---------|--------|
| OpenAI | `embedders/openai` | text-embedding-3-small (1536d), text-embedding-3-large (3072d) |
| Ollama | `embedders/ollama` | nomic-embed-text, mxbai-embed-large, all-minilm |
| Cohere | `embedders/cohere` | embed-v4.0 (1024d) |
| Gemini | `embedders/gemini` | text-embedding-004 (768d) |
| Voyage AI | `embedders/voyage` | voyage-3 (1024d) |

## Vector Stores

| Store | Package | Use Case |
|-------|---------|----------|
| In-Memory | `stores/inmemory` | Development, testing, small scale (< 10K) |
| PostgreSQL + pgvector | `stores/pgvector` | Production, HNSW indexing, GORM |
| SQLite | `stores/sqlite` | Edge, CLI tools, single-machine |
| Qdrant | `stores/qdrant` | High-performance vector DB, REST API |

## Documentation

| | |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation and quick start |
| [Embedders](docs/embedders.md) | Embedding provider configuration |
| [Stores](docs/stores.md) | Vector store backends |
| [Scoring](docs/scoring.md) | Importance decay and retrieval ranking |

Full docs: [promptrails.github.io/memoryrails](https://promptrails.github.io/memoryrails)

## Part of the PromptRails AI Toolkit

- [LangRails](https://github.com/promptrails/langrails) — Unified LLM provider interface
- [GuardRails](https://github.com/promptrails/guardrails) — Content safety scanning
- **MemoryRails** — Agent memory management
- [MediaRails](https://github.com/promptrails/mediarails) — AI media generation

## License

MIT — [PromptRails](https://promptrails.com)
