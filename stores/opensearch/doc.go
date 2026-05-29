// Package opensearch provides an Amazon OpenSearch vector store for memoryrails.
//
// It works with both OpenSearch Serverless (service "aoss", the default) and
// managed OpenSearch domains (service "es"), using k-NN search over the REST
// API. Requests are signed with AWS Signature V4 using only the standard
// library (no aws-sdk-go dependency).
//
//	store, err := opensearch.New(
//		"https://abc123.us-east-1.aoss.amazonaws.com",
//		"memories",
//		1024, // embedding dimensions
//		opensearch.WithRegion("us-east-1"),
//	)
//
// The index is created on first use with a knn_vector mapping. For managed
// domains pass opensearch.WithService("es").
//
// Note: Search results' Similarity carries OpenSearch's normalized _score for
// the configured k-NN space type, not a raw cosine value.
package opensearch
