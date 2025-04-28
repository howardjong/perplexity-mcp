
# Perplexity MCP Server

This is a Python implementation of the Model Context Protocol (MCP) server for Perplexity. The server is built using FastAPI and follows the MCP specification.

## Features

- Implements the core MCP endpoints:
  - `/v1/models` - List available models registered in the local MCP server
  - `/v1/models/{model_id}` - Get model information for a locally registered model
  - `/v1/models/{model_id}/chat` - Chat with a model - this endpoint actually calls the Perplexity API
  - `/v1/models/{model_id}/complete` - Text completion (demo implementation)
- Additional utility endpoints:
  - `/perplexity-models` - Lists the known models available in the Perplexity API
  - `/api-key-test` - Test if your Perplexity API key is properly configured
  - `/server-info` - Get information about the server
  - `/health` - Health check endpoint
- Ready for deployment

## Running the Server

### Locally

1. Create a `.env` file in the project root with your Perplexity API key:
   ```env
   PERPLEXITY_API_KEY=your_actual_key_here
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python main.py
   ```
The server will start on http://0.0.0.0:5000 (or the next available port).

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t perplexity-mcp .
   ```
2. Run the container, passing your `.env` file:
   ```bash
   docker run --env-file .env -p 5003:5000 perplexity-mcp
   ```
   - The API will be available at http://localhost:5003
   - You can use any available port on your host (replace 5003 as needed)

**Note:**
- `.env` is included in `.gitignore` and `.dockerignore` to keep your API key secure.
- Never commit your `.env` file to GitHub.

### Render Deployment

Render can automatically build and deploy this project from GitHub using the included Dockerfile.

1. Push your code to GitHub.
2. In the Render dashboard, create a new Web Service:
   - Select "Docker" as the environment.
   - Connect your GitHub repo and select the branch (e.g., `main`).
   - Set your environment variable `PERPLEXITY_API_KEY` in the Render dashboard (Settings > Environment).
   - Choose your desired port (default is 5000).
3. Deploy! Render will build and run your Docker container.

**Security:** Do not rely on `.env` in the repo for Render. Always set secrets in the Render dashboard.

## API Documentation

Once the server is running, you can access the API documentation at `/docs` (e.g., http://0.0.0.0:5000/docs).

## Model Registry

The server includes a simple in-memory model registry. In a production environment, you'd want to replace this with a more sophisticated registry system.

## Adding Your Own Models

To add your own models, update the `MODEL_REGISTRY` dictionary in main.py with your model configurations.

## Supported Models

The server supports the following Perplexity models:

- `sonar` - Perplexity's flagship model with strong reasoning
- `sonar-small` - Smaller, faster version of Sonar
- `sonar-medium` - Medium-sized version of Sonar
- `sonar-pro` - Pro version of Sonar with enhanced capabilities
- `sonar-deep-research` - Specialized for in-depth research tasks
- `sonar-reasoning-pro` - Advanced reasoning capabilities with enhanced logic
- `codellama-70b` - Specialized for code generation
- `mixtral-8x7b` - From Mistral AI, good for general tasks
- `mistral-7b` - Fast and efficient model from Mistral AI

To use a specific model, simply call `/v1/models/{model_id}/chat` with the model ID.

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
