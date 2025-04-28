import os
import json
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# Use local implementation instead of the problematic package
try:
    from modelcontextprotocol import ModelResponse, ChatMessage, ToolCall, MessageRole, ToolResult
    print("Successfully imported modelcontextprotocol")
except ImportError:
    print("Using local implementation of MCP classes")
    from local_mcp import ModelResponse, ChatMessage, ToolCall, MessageRole, ToolResult

app = FastAPI(title="Perplexity MCP Server")

# Model configuration
class ModelConfig(BaseModel):
    model_id: str = "demo-model"
    display_name: str = "Demo Model"
    description: str = "A demonstration MCP-compatible model"
    capabilities: List[str] = ["chat"]
    max_input_tokens: int = 4096
    max_total_tokens: int = 8192

# Simple in-memory model registry
# In a production environment, you'd likely have a more sophisticated registry
MODEL_REGISTRY = {
    "demo-model": ModelConfig()
}

@app.get("/")
async def root():
    return {"message": "Perplexity MCP Server is running"}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = []
    for model_id, config in MODEL_REGISTRY.items():
        models.append({
            "id": model_id,
            "display_name": config.display_name,
            "description": config.description,
            "capabilities": config.capabilities,
            "max_input_tokens": config.max_input_tokens,
            "max_total_tokens": config.max_total_tokens
        })
    return {"models": models}

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get model information"""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    config = MODEL_REGISTRY[model_id]
    return {
        "id": model_id,
        "display_name": config.display_name,
        "description": config.description,
        "capabilities": config.capabilities,
        "max_input_tokens": config.max_input_tokens,
        "max_total_tokens": config.max_total_tokens
    }

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature between 0 and 2")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter between 0 and 1")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty between -2 and 2")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty between -2 and 2")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences that cause the model to stop generating")
    repetition_penalty: Optional[float] = Field(default=None, description="Repetition penalty for token generation")
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens")
    tool_results: Optional[List[ToolResult]] = None

@app.post("/v1/models/{model_id}/chat")
async def chat(model_id: str, request: ChatRequest):
    """Chat with the model"""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Mock response generation
    # In a real implementation, you would call your model here with all parameters
    system_message = "I am a helpful assistant."
    user_messages = [msg for msg in request.messages if msg.role == MessageRole.USER]

    if not user_messages:
        return ModelResponse(
            content="I don't see any user messages. How can I help you today?",
            role=MessageRole.ASSISTANT,
            tool_calls=[]
        )

    last_user_message = user_messages[-1].content

    # Log parameters (in a real implementation, these would be passed to the model)
    params = {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "stop": request.stop,
        "repetition_penalty": request.repetition_penalty,
        "logprobs": request.logprobs,
        "stream": request.stream
    }
    print(f"Model parameters: {params}")

    # Simple demo response
    response = f"You said: {last_user_message}. This is a demo response from the MCP server."

    return ModelResponse(
        content=response,
        role=MessageRole.ASSISTANT,
        tool_calls=[]
    )

@app.post("/v1/models/{model_id}/complete")
async def complete(model_id: str, request: Dict[str, Any]):
    """Text completion endpoint"""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    if "complete" not in MODEL_REGISTRY[model_id].capabilities:
        raise HTTPException(status_code=400, detail=f"Model {model_id} does not support completion")

    # Mock completion implementation
    prompt = request.get("prompt", "")
    response = f"Completion for: {prompt}. This is a demo completion from the MCP server."

    return {
        "text": response,
        "finish_reason": "stop"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Try to find an available port starting from 5000
    port = int(os.environ.get("PORT", 5000))
    max_port_attempts = 10
    
    for attempt in range(max_port_attempts):
        try:
            # Try to create a socket to check if the port is available
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            # If we get here, the port is available
            break
        except socket.error:
            print(f"Port {port} is already in use, trying {port + 1}")
            port += 1
            if attempt == max_port_attempts - 1:
                print(f"Could not find an available port after {max_port_attempts} attempts")
                import sys
                sys.exit(1)
    
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)