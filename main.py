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

    api_key = os.environ.get("PERPLEXITY_API_KEY")

    # Check if API key is available (when connecting to actual Perplexity API)
    if not api_key:
        print("Warning: No Perplexity API key found. Using demo implementation.")
        # Continue with demo implementation below
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

    # This is the path with the API key
    print(f"Using Perplexity API key: {api_key[:5]}...")
    # Make an actual call to the Perplexity API
    import requests
    
    perplexity_url = "https://api.perplexity.ai/chat/completions"
    
    # Convert MCP messages to Perplexity format
    perplexity_messages = [
        {"role": msg.role.value, "content": msg.content} for msg in request.messages
    ]
    
    # Prepare request payload - using simpler format as shown in example
    payload = {
        "model": "sonar",  # Use one of Perplexity's current models
        "messages": perplexity_messages
    }
    
    # Only add optional parameters if specified
    if request.max_tokens:
        payload["max_tokens"] = request.max_tokens
    if request.temperature:
        payload["temperature"] = request.temperature
    if request.top_p:
        payload["top_p"] = request.top_p
    # Stream is always false since we're not handling streaming
    payload["stream"] = False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending request to Perplexity API: {perplexity_url}")
        print(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(perplexity_url, headers=headers, json=payload)
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"API response: {json.dumps(result, indent=2)}")
                
                # Perplexity API standard format has 'choices' array
                if not result.get("choices"):
                    print("WARNING: 'choices' field is missing or empty in the response")
                    return ModelResponse(
                        content=f"Invalid response from Perplexity API: {response.text[:200]}",
                        role=MessageRole.ASSISTANT,
                        tool_calls=[]
                    )
                
                assistant_message = result.get("choices", [{}])[0].get("message", {})
                if not assistant_message:
                    print("WARNING: 'message' field is missing in the first choice")
                    return ModelResponse(
                        content=f"Invalid response format from Perplexity API: {response.text[:200]}",
                        role=MessageRole.ASSISTANT,
                        tool_calls=[]
                    )
                
                content = assistant_message.get("content")
                if content is None:
                    print("WARNING: 'content' field is null or missing in the message")
                    # Return full response for debugging
                    return ModelResponse(
                        content=f"Perplexity API returned a null content field. Full response: {response.text[:500]}",
                        role=MessageRole.ASSISTANT,
                        tool_calls=[]
                    )
                
                return ModelResponse(
                    content=content,
                    role=MessageRole.ASSISTANT,
                    tool_calls=[]
                )
            except json.JSONDecodeError as e:
                print(f"Failed to parse response as JSON: {e}")
                return ModelResponse(
                    content=f"Failed to parse response as JSON: {str(e)}. Raw response: {response.text[:200]}",
                    role=MessageRole.ASSISTANT,
                    tool_calls=[]
                )
        else:
            error_msg = f"Perplexity API error: {response.status_code} - {response.text}"
            print(error_msg)
            return ModelResponse(
                content=f"Error calling Perplexity API: {response.status_code} - {response.text[:200]}",
                role=MessageRole.ASSISTANT,
                tool_calls=[]
            )
    except Exception as e:
        error_msg = f"Exception when calling Perplexity API: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return ModelResponse(
            content=f"Error: {str(e)}",
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