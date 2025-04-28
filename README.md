
# Perplexity MCP Server

This is a Python implementation of the Model Context Protocol (MCP) server for Perplexity. The server is built using FastAPI and follows the MCP specification.

## Features

- Implements the core MCP endpoints:
  - `/v1/models` - List available models
  - `/v1/models/{model_id}` - Get model information
  - `/v1/models/{model_id}/chat` - Chat with a model
  - `/v1/models/{model_id}/complete` - Text completion (demo implementation)
- Includes a health check endpoint
- Ready for deployment

## Running the Server

### Locally

```bash
python main.py
```

The server will start on http://0.0.0.0:5000

### Deployment on Replit

This server is configured to be deployed on Replit. The deployment will use the main.py file as the entry point.

## API Documentation

Once the server is running, you can access the API documentation at `/docs` (e.g., http://0.0.0.0:5000/docs).

## Model Registry

The server includes a simple in-memory model registry. In a production environment, you'd want to replace this with a more sophisticated registry system.

## Adding Your Own Models

To add your own models, update the `MODEL_REGISTRY` dictionary in main.py with your model configurations.

## Supported Model Parameters

The chat endpoint supports the following parameters from the Perplexity API:

- `max_tokens`: Maximum number of tokens to generate
- `temperature`: Sampling temperature between 0 and 2
- `top_p`: Nucleus sampling parameter between 0 and 1
- `top_k`: Top-k sampling parameter
- `presence_penalty`: Presence penalty between -2 and 2
- `frequency_penalty`: Frequency penalty between -2 and 2
- `stop`: Stop sequences that cause the model to stop generating
- `repetition_penalty`: Repetition penalty for token generation
- `logprobs`: Whether to return log probabilities of the output tokens
- `stream`: Whether to stream the response
