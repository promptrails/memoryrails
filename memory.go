package memoryrails

import "time"

// MemoryType classifies the kind of memory stored.
type MemoryType string

const (
	// TypeConversation stores conversational exchanges.
	TypeConversation MemoryType = "conversation"

	// TypeFact stores factual information (e.g., "user lives in Istanbul").
	TypeFact MemoryType = "fact"

	// TypeProcedure stores procedural knowledge (e.g., "to deploy, run X then Y").
	TypeProcedure MemoryType = "procedure"

	// TypeEpisodic stores event-based memories (e.g., "user reported a bug on March 5").
	TypeEpisodic MemoryType = "episodic"

	// TypeSemantic stores contextual/semantic information.
	TypeSemantic MemoryType = "semantic"
)

// Memory is a single memory entry with embedding and metadata.
type Memory struct {
	// ID is the unique identifier.
	ID string

	// Content is the text content of the memory.
	Content string

	// Type classifies this memory.
	Type MemoryType

	// Embedding is the vector representation of the content.
	Embedding []float32

	// Metadata holds arbitrary key-value data.
	Metadata map[string]any

	// Importance is a score from 0 to 1 indicating how important this memory is.
	// Higher values are prioritized in retrieval. Subject to time-based decay.
	Importance float64

	// AccessCount tracks how many times this memory has been retrieved.
	AccessCount int

	// LastAccessedAt is the last time this memory was retrieved.
	LastAccessedAt time.Time

	// CreatedAt is when the memory was created.
	CreatedAt time.Time

	// UpdatedAt is when the memory was last modified.
	UpdatedAt time.Time
}
