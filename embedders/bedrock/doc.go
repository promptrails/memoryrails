// Package bedrock provides an Amazon Bedrock embedder for memoryrails.
//
// It supports the Amazon Titan and Cohere embedding model families via
// Bedrock's InvokeModel API and signs requests with AWS Signature V4 using only
// the standard library (no aws-sdk-go dependency).
//
// Credentials and region default to the standard AWS environment variables
// (AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN):
//
//	emb := bedrock.New(
//		bedrock.WithRegion("us-east-1"),
//		bedrock.WithModel(bedrock.ModelTitanV2),
//		bedrock.WithDimensions(512), // Titan V2 only: 256, 512 or 1024
//	)
//	vec, err := emb.Embed(ctx, "hello world")
//
// Titan models embed one input per request; Cohere models embed a whole batch
// in a single request.
package bedrock
