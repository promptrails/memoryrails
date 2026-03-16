# Embedding Providers

All embedders implement the `Embedder` interface:

```go
type Embedder interface {
    Embed(ctx context.Context, text string) ([]float32, error)
    EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
    Dimensions() int
}
```

## OpenAI

```go
import "github.com/promptrails/memoryrails/embedders/openai"

e := openai.New("sk-...")

// Large model
e := openai.New("sk-...", openai.WithModel(openai.ModelLarge))
```

| Model | Dimensions | Cost |
|-------|-----------|------|
| text-embedding-3-small | 1536 | Lower |
| text-embedding-3-large | 3072 | Higher |

## Ollama (Local)

```go
import "github.com/promptrails/memoryrails/embedders/ollama"

e := ollama.New() // defaults: localhost:11434, nomic-embed-text

// Custom model
e := ollama.New(
    ollama.WithModel("mxbai-embed-large"),
    ollama.WithDimensions(1024),
)

// Remote Ollama
e := ollama.New(ollama.WithBaseURL("http://gpu-server:11434"))
```

## Cohere

```go
import "github.com/promptrails/memoryrails/embedders/cohere"

e := cohere.New("your-api-key") // embed-v4.0, 1024d
```

## Google Gemini

```go
import "github.com/promptrails/memoryrails/embedders/gemini"

e := gemini.New("your-api-key") // text-embedding-004, 768d
```

## Voyage AI

```go
import "github.com/promptrails/memoryrails/embedders/voyage"

e := voyage.New("your-api-key") // voyage-3, 1024d
```

## Custom Embedder

Implement the `Embedder` interface:

```go
type MyEmbedder struct{}

func (e *MyEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
    // Your embedding logic
    return vector, nil
}

func (e *MyEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
    results := make([][]float32, len(texts))
    for i, t := range texts {
        emb, err := e.Embed(ctx, t)
        if err != nil { return nil, err }
        results[i] = emb
    }
    return results, nil
}

func (e *MyEmbedder) Dimensions() int { return 384 }
```
