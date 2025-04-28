
import os
import json
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import requests  # Ensure requests is imported
import traceback  # For better exception logging

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
    display_name: str = "Demo Model (Proxy for Perplexity)"
    description: str = "A demonstration MCP-compatible server proxying to Perplexity API"
    capabilities: List[str] = ["chat"]  # Only implementing chat for now
    max_input_tokens: int = 4096
    max_total_tokens: int = 8192

# Simple in-memory model registry
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

    api_key = os.environ.get("PERPLEXITY_API_KEY")

    # Check if API key is available (when connecting to actual Perplexity API)
    if not api_key:
        print("Warning: No Perplexity API key found. Using demo implementation.")
        # Continue with demo implementation below
        user_messages = [msg for msg in request.messages if msg.role == MessageRole.USER]

        if not user_messages:
            return ModelResponse(
                content="I don't see any user messages. How can I help you today?",
                role=MessageRole.ASSISTANT,
                tool_calls=[]
            )

        last_user_message = user_messages[-1].content

        # Log parameters for demo implementation
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
    else:
        # Perplexity API implementation
        print(f"Using Perplexity API key: {api_key[:4]}...{api_key[-4:]}")
        
        # Choose a specific Perplexity model
        perplexity_model_name = "sonar-medium-online"  # You can change this to another model
        print(f"Targeting Perplexity model: {perplexity_model_name}")
        
        perplexity_url = "https://api.perplexity.ai/chat/completions"
        
        # Convert MCP messages to Perplexity format
        perplexity_messages = []
        for msg in request.messages:
            try:
                # Ensure role is the string value (e.g., "user", "assistant")
                role_value = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                perplexity_messages.append({"role": role_value, "content": msg.content})
            except AttributeError:
                print(f"Warning: Could not get '.value' from message role: {msg.role}. Using raw value.")
                perplexity_messages.append({"role": str(msg.role), "content": msg.content})
        
        # Prepare request payload
        payload = {
            "model": perplexity_model_name,
            "messages": perplexity_messages
        }
        
        # Add optional parameters if provided
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.stop is not None:
            payload["stop"] = request.stop
            
        # MCP stream=True not handled here
        payload["stream"] = False
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            print(f"Sending request to Perplexity API: {perplexity_url}")
            print(f"Request payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(perplexity_url, headers=headers, json=payload, timeout=120)
            
            print(f"Response status code: {response.status_code}")
            
            response.raise_for_status()  # Raise HTTPError for bad responses
            
            result = response.json()
            
            # Check for 'choices' array
            if not result.get("choices"):
                error_msg = "ERROR: 'choices' field is missing or empty in the Perplexity API response."
                print(error_msg)
                raise HTTPException(status_code=500, detail=f"Invalid response from Perplexity API: 'choices' missing. Response: {response.text[:200]}")
            
            first_choice = result["choices"][0]
            assistant_message = first_choice.get("message")
            
            if not assistant_message:
                error_msg = "ERROR: 'message' field is missing in the first choice of the Perplexity API response."
                print(error_msg)
                raise HTTPException(status_code=500, detail=f"Invalid response format from Perplexity API: 'message' missing. Response: {response.text[:200]}")
            
            content = assistant_message.get("content")
            
            if content is None:
                print("WARNING: 'content' field is null or missing in the Perplexity API message.")
                content = ""  # Set empty content if null
            
            return ModelResponse(
                content=content,
                role=MessageRole.ASSISTANT,
                tool_calls=[]
            )
            
        except requests.exceptions.Timeout:
            error_msg = "ERROR: Request to Perplexity API timed out."
            print(error_msg)
            raise HTTPException(status_code=504, detail="Request to upstream API timed out")
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Perplexity API: {e}"
            print(error_msg)
            error_detail = error_msg
            if hasattr(e, 'response') and e.response is not None:
                error_detail = f"Perplexity API Error: {e.response.status_code} - {e.response.text[:200]}"
            raise HTTPException(status_code=502, detail=error_detail)
        except json.JSONDecodeError as e:
            error_msg = f"ERROR: Failed to parse JSON response from Perplexity API: {e}"
            print(error_msg)
            if 'response' in locals():
                print(f"Raw response that failed parsing: {response.text[:500]}")
                raise HTTPException(status_code=500, detail=f"Failed to parse upstream API response as JSON. Raw response: {response.text[:200]}")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to parse upstream API response as JSON: {str(e)}")
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            print(error_msg)
            traceback.print_exc()  # Print full traceback for debugging
            raise HTTPException(status_code=500, detail=error_msg)

@app.post("/v1/models/{model_id}/complete")
async def complete(model_id: str, request: Dict[str, Any]):
    """Text completion endpoint"""
    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Only perform capability check if completion capability is listed
    if "complete" in MODEL_REGISTRY[model_id].capabilities and "complete" not in MODEL_REGISTRY[model_id].capabilities:
        raise HTTPException(status_code=400, detail=f"Model {model_id} does not support completion")

    # Mock completion implementation
    prompt = request.get("prompt", "")
    response = f"Completion for: {prompt}. This is a demo completion from the MCP server."

    return {
        "choices": [{
            "text": response,
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop"
        }],
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api-key-test")
async def test_api_key():
    """Test if the Perplexity API key is properly set"""
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    
    if not api_key:
        return {"status": "error", "message": "No API key found"}
    else:
        # Only show first few characters for security
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        return {"status": "success", "message": f"API key found: {masked_key}"}

@app.get("/server-info")
async def server_info():
    """Server information endpoint"""
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    # Get the port from the current server
    port = 5000  # Default value
    return {
        "status": "running",
        "hostname": hostname,
        "local_ip": local_ip,
        "server_port": port,
        "external_port_mapping": "5000 -> 80",
        "api_key_configured": bool(os.environ.get("PERPLEXITY_API_KEY"))
    }

if __name__ == "__main__":
    import uvicorn
    import socket

    # Try to find an available port starting from 5000
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")  # Allow configuring host
    max_port_attempts = 10
    found_port = False

    for attempt in range(max_port_attempts):
        try:
            # Try to create a socket to check if the port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
            # If we get here, the port is available
            found_port = True
            break
        except OSError:  # Catch OSError which includes 'address already in use'
            print(f"Port {port} on host {host} is already in use, trying {port + 1}")
            port += 1

    if not found_port:
        print(f"Could not find an available port after {max_port_attempts} attempts starting from {port - max_port_attempts}")
        import sys
        sys.exit(1)

    print(f"Starting server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
