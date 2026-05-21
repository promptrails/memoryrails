// Package openai provides an OpenAI embedding provider for memoryrails.
//
// Supports text-embedding-3-small (1536 dims) and text-embedding-3-large (3072 dims).
// Both 3-* models support shorter output vectors via OpenAI's `dimensions`
// request parameter; configure via WithDimensions.
package openai
