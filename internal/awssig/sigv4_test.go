package awssig

import (
	"net/http"
	"regexp"
	"strings"
	"testing"
	"time"
)

func testSigner(serviceName string) *Signer {
	return &Signer{
		Credentials: Credentials{
			AccessKeyID:     "AKIDEXAMPLE",
			SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
		},
		Region:  "us-east-1",
		Service: serviceName,
	}
}

func TestSign_SetsRequiredHeaders(t *testing.T) {
	s := testSigner("aoss")
	req, _ := http.NewRequest(http.MethodPost, "https://abc.us-east-1.aoss.amazonaws.com/idx/_search", nil)
	s.Sign(req, []byte(`{"q":1}`), time.Unix(1700000000, 0))

	if len(req.Header.Get("X-Amz-Content-Sha256")) != 64 {
		t.Error("missing/short X-Amz-Content-Sha256")
	}
	auth := req.Header.Get("Authorization")
	if !strings.Contains(auth, "/us-east-1/aoss/aws4_request") {
		t.Errorf("scope wrong: %q", auth)
	}
	if !regexp.MustCompile(`Signature=[0-9a-f]{64}$`).MatchString(auth) {
		t.Errorf("signature malformed: %q", auth)
	}
}

func TestSign_Deterministic(t *testing.T) {
	s := testSigner("bedrock")
	at := time.Unix(1700000000, 0)
	body := []byte(`{"inputText":"hi"}`)

	r1, _ := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke", nil)
	r2, _ := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke", nil)
	s.Sign(r1, body, at)
	s.Sign(r2, body, at)

	if r1.Header.Get("Authorization") != r2.Header.Get("Authorization") {
		t.Error("signing not deterministic")
	}

	r3, _ := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/x/invoke", nil)
	s.Sign(r3, []byte(`{"inputText":"different"}`), at)
	if r3.Header.Get("Authorization") == r1.Header.Get("Authorization") {
		t.Error("signature should change with body")
	}
}
