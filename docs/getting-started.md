# Getting Started

## Installation

```bash
go get github.com/promptrails/memoryrails
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/promptrails/memoryrails"
    "github.com/promptrails/memoryrails/embedders/openai"
    "github.com/promptrails/memoryrails/stores/inmemory"
)

func main() {
    // Create embedder and store
    embedder := openai.New("sk-...")
    store := inmemory.New()

    // Create manager
    mgr := memoryrails.NewManager(embedder, store)
    ctx := context.Background()

    // Store memories
    mgr.Remember(ctx, "User's name is Alice", memoryrails.TypeFact, nil)
    mgr.Remember(ctx, "User prefers dark mode", memoryrails.TypeFact, nil)
    mgr.Remember(ctx, "Deployed v2.0 on March 5", memoryrails.TypeEpisodic, nil)

    // Recall relevant memories
    results, _ := mgr.Recall(ctx, "What do I know about the user?", memoryrails.RecallOptions{
        Limit: 5,
    })

    for _, r := range results {
        fmt.Printf("[%.2f] %s\n", r.Similarity, r.Memory.Content)
    }
}
```

## Memory Types

| Type | Use Case |
|------|----------|
| `TypeConversation` | Conversational exchanges |
| `TypeFact` | Factual information ("user lives in Istanbul") |
| `TypeProcedure` | Procedural knowledge ("to deploy, run X then Y") |
| `TypeEpisodic` | Event-based memories ("bug reported on March 5") |
| `TypeSemantic` | Contextual/semantic information |

## With LangRails

```go
import (
    "github.com/promptrails/langrails"
    oaiProvider "github.com/promptrails/langrails/openai"
    "github.com/promptrails/memoryrails"
    oaiEmbed "github.com/promptrails/memoryrails/embedders/openai"
    "github.com/promptrails/memoryrails/stores/inmemory"
)

// LLM provider
provider := oaiProvider.New("sk-...")

// Memory manager
mgr := memoryrails.NewManager(oaiEmbed.New("sk-..."), inmemory.New())

// Store user preference
mgr.Remember(ctx, "User prefers concise answers", memoryrails.TypeFact, nil)

// Before each LLM call, recall relevant memories
results, _ := mgr.Recall(ctx, userInput, memoryrails.RecallOptions{Limit: 3})

// Build context from memories
var memoryContext string
for _, r := range results {
    memoryContext += "- " + r.Memory.Content + "\n"
}

// Use in prompt
resp, _ := provider.Complete(ctx, &langrails.CompletionRequest{
    Model: "gpt-4o",
    SystemPrompt: "You are a helpful assistant.\n\nRelevant context:\n" + memoryContext,
    Messages: []langrails.Message{{Role: "user", Content: userInput}},
})
```

## Custom Importance

```go
importance := 0.9
mgr.Remember(ctx, "Critical: user is allergic to peanuts", memoryrails.TypeFact, nil,
    memoryrails.RememberOptions{Importance: &importance},
)
```

## Next Steps

- [Embedders](embedders.md) — Configure embedding providers
- [Stores](stores.md) — Vector store options
- [Scoring](scoring.md) — Importance decay and ranking
