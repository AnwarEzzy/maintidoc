# Local language model setup

MaintiDoc uses Ollama to run Qwen locally.

## Selected model

- Model: qwen3:4b-instruct-2507-q4_K_M
- Role: generate answers from retrieved document passages.
- API endpoint: http://127.0.0.1:11434/api/chat

The Instruct variant is selected for direct answers.
The generic qwen3:4b tag tested previously produced reasoning
text despite the think=false setting.

## Installation

Run: ollama pull qwen3:4b-instruct-2507-q4_K_M

## Initial validation

The chat test requests the sentence "MaintiDoc is ready."

Test settings:
- Context window: 2048 tokens.
- Maximum generated tokens: 128.
- Streaming disabled.
- keep_alive set to 0 to unload the model after the response.

Expected result:
- message.content contains the requested sentence.
- done_reason is stop.

This test validates local generation.
Connecting retrieved passages and validating citations are next.