package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	os.Setenv("OPENAI_API_KEY", "ollama")
	defer func() {
		os.Unsetenv("OPENAI_API_KEY")
	}()

	llm, err := openai.New(
		openai.WithBaseURL("http://localhost:11434/v1"),
		openai.WithModel("devstral"),
	)
	if err != nil {
		log.Fatalf("failed to create LLM: %v", err)
	}
	agentTools := []tools.Tool{
		AddTool{},
		ReadFileTool{},
	}

	agent := agents.NewOneShotAgent(llm,
		agentTools,
		agents.NewOpenAIOption().WithSystemMessage("You are a helpful assistant."),
		agents.NewOpenAIOption().WithSystemMessage(`You can answer math questions using available tools.`),
		agents.NewOpenAIOption().WithSystemMessage("You can read local files with the tool 'read_file'."),
		agents.WithMaxIterations(3),
		// agents.WithCallbacksHandler(callbacks.LogHandler{}),
	)
	executor := agents.NewExecutor(agent)

	question := "What does the file 'main.go' do? Tell me the basics"
	// question := "What is 12 + 30?"
	answer, err := chains.Run(context.Background(), executor, question)
	fmt.Println(answer)
	return err
}

type AddTool struct{}

func (t AddTool) Name() string {
	return "add"
}

func (t AddTool) Description() string {
	return "Add two numbers. Input format: 'number1 number2' (space separated)"
}

func (t AddTool) Call(ctx context.Context, input string) (string, error) {
	input = strings.Trim(input, `"'`)
	input = strings.TrimSpace(input)
	parts := strings.Fields(input)

	if len(parts) != 2 {
		return "", fmt.Errorf("expected 2 numbers, got %d", len(parts))
	}

	a, err := strconv.ParseFloat(parts[0], 64)
	if err != nil {
		return "", fmt.Errorf("invalid first number: %v", err)
	}

	b, err := strconv.ParseFloat(parts[1], 64)
	if err != nil {
		return "", fmt.Errorf("invalid second number: %v", err)
	}

	result := a + b
	return fmt.Sprintf("%.0f", result), nil
}

type ReadFileTool struct{}

func (t ReadFileTool) Name() string {
	return "read_file"
}

func (t ReadFileTool) Description() string {
	return "Read and return the contents of a text file. Provide the relative or absolute path to the file."
}

func (t ReadFileTool) Call(ctx context.Context, input string) (string, error) {
	pathVal := filepath.Clean(input)

	// Check if file exists
	if _, err := os.Stat(pathVal); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", pathVal)
	}

	// Read the file
	content, err := os.ReadFile(pathVal)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %w", pathVal, err)
	}

	// Return with some metadata for debugging
	return fmt.Sprintf("File: %s\nContent:\n%s", pathVal, string(content)), nil
}
