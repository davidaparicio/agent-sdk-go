package tracing

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type errorStreamLLM struct {
	streamErr error
}

func (*errorStreamLLM) Generate(context.Context, string, ...interfaces.GenerateOption) (string, error) {
	return "", nil
}

func (*errorStreamLLM) GenerateWithTools(context.Context, string, []interfaces.Tool, ...interfaces.GenerateOption) (string, error) {
	return "", nil
}

func (*errorStreamLLM) GenerateDetailed(context.Context, string, ...interfaces.GenerateOption) (*interfaces.LLMResponse, error) {
	return &interfaces.LLMResponse{}, nil
}

func (*errorStreamLLM) GenerateWithToolsDetailed(context.Context, string, []interfaces.Tool, ...interfaces.GenerateOption) (*interfaces.LLMResponse, error) {
	return &interfaces.LLMResponse{}, nil
}

func (*errorStreamLLM) GenerateStream(context.Context, string, ...interfaces.GenerateOption) (<-chan interfaces.StreamEvent, error) {
	return nil, nil
}

func (m *errorStreamLLM) GenerateWithToolsStream(ctx context.Context, _ string, _ []interfaces.Tool, _ ...interfaces.GenerateOption) (<-chan interfaces.StreamEvent, error) {
	AddToolCallToContext(ctx, ToolCall{Name: "test_tool", Timestamp: time.Now().Format(time.RFC3339)})
	ch := make(chan interfaces.StreamEvent, 1)
	if m.streamErr != nil {
		ch <- interfaces.StreamEvent{Type: interfaces.StreamEventError, Error: m.streamErr}
	} else {
		ch <- interfaces.StreamEvent{Type: interfaces.StreamEventContentComplete, Content: "complete"}
	}
	close(ch)
	return ch, nil
}

func (*errorStreamLLM) Name() string            { return "test" }
func (*errorStreamLLM) SupportsStreaming() bool { return true }

func TestGenerateWithToolsStreamGenerationSpan(t *testing.T) {
	for _, tc := range []struct {
		name           string
		includeContent bool
		streamErr      error
		wantError      string
		wantResponse   string
	}{
		{
			name:           "error with content included",
			includeContent: true,
			streamErr:      errors.New("stream interrupted"),
			wantError:      "stream interrupted",
			wantResponse:   "<error>",
		},
		{
			name:           "error with content redacted",
			includeContent: false,
			streamErr:      errors.New("stream interrupted"),
			wantError:      "stream interrupted",
			wantResponse:   "<redacted>",
		},
		{
			name:           "successful stream preserves response placeholder",
			includeContent: false,
			wantResponse:   "streaming_response",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			exporter := tracetest.NewInMemoryExporter()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			t.Cleanup(func() { _ = provider.Shutdown(context.Background()) })

			otelTracer := &OTELLangfuseTracer{
				tracerProvider: provider,
				tracer:         provider.Tracer("test"),
				enabled:        true,
				IncludeContent: tc.includeContent,
			}
			traced := NewTracedLLM(&errorStreamLLM{streamErr: tc.streamErr}, NewOTELTracerAdapter(otelTracer)).(*TracedLLM)

			stream, err := traced.GenerateWithToolsStream(context.Background(), "secret prompt", nil)
			if err != nil {
				t.Fatalf("GenerateWithToolsStream() error = %v", err)
			}
			for range stream {
			}

			var generation sdktrace.ReadOnlySpan
			for _, span := range exporter.GetSpans() {
				for _, attr := range span.Attributes {
					if string(attr.Key) == "langfuse.observation.type" && attr.Value.AsString() == "generation" {
						generation = span.Snapshot()
					}
				}
			}
			if generation == nil {
				t.Fatal("generation span was not exported")
			}

			attributes := make(map[string]string)
			for _, attr := range generation.Attributes() {
				attributes[string(attr.Key)] = attr.Value.AsString()
			}
			if got := attributes["langfuse.observation.metadata.error"]; got != tc.wantError {
				t.Errorf("error metadata = %q, want %q", got, tc.wantError)
			}
			if got := attributes["gen_ai.completion.0.content"]; got != tc.wantResponse {
				t.Errorf("generation response = %q, want %q", got, tc.wantResponse)
			}

			if tc.streamErr != nil {
				var foundErrorEvent bool
				for _, span := range exporter.GetSpans() {
					if span.Name != "llm.generate_with_tools_stream" {
						continue
					}
					for _, event := range span.Events {
						if event.Name == "exception" {
							foundErrorEvent = true
						}
					}
				}
				if !foundErrorEvent {
					t.Error("main streaming span did not record the stream error")
				}
			}
		})
	}
}
