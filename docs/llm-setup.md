# Local language model setup

MaintiDoc uses Ollama to run Qwen locally.

## Selected model

- Model: qwen3:4b-instruct-2507-q4_K_M
- Role: route searches, collect evidence and generate answers from retrieved passages.
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
The RAG and LangChain agent now connect retrieved passages and validate citations.

## Vision model

- Model: `qwen3-vl:2b-instruct-q4_K_M`.
- Role: describe extracted figures/table previews and observe images for the
  automatic caption consistency filter.
- Installation: `ollama pull qwen3-vl:2b-instruct-q4_K_M`.

If Ollama rejects the model with HTTP 412, update Ollama and restart its
application before retrying. Do not disable TLS verification to work around
download or certificate errors.

## Current runtime settings

The LangChain agent uses an 8192-token context window, at most 768 generated
tokens per model call and `keep_alive=0`. The original CLI RAG uses its smaller
4096-token context. The vision helper also uses a 4096-token context and checks
that generation finished normally before accepting any description.

See [the LangChain guide](langchain-guide.md) for the distinction between tool
selection, evidence collection and the separate grounded generation call.
