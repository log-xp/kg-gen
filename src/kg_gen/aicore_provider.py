"""SAP AI Core provider for DSPy."""

from typing import Any, Dict, List, Optional
import dspy
from dspy.clients.lm import LM


class AICoreLM(LM):
    """
    A DSPy LM wrapper for SAP AI Core that uses the gen-ai-hub-sdk.
    
    This allows kg-gen to work with SAP AI Core's LLM proxy service instead of directly calling OpenAI.
    
    Example usage:
        from gen_ai_hub.proxy.native.amazon.clients import Session
        from kg_gen.aicore_provider import AICoreLM
        from kg_gen import KGGen
        
        # Create SAP AI Core client
        bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
        
        # Create AICoreLM wrapper
        lm = AICoreLM(client=bedrock, model_name="anthropic--claude-4-sonnet")
        
        # Use with KGGen
        kg = KGGen(aicore_client=bedrock, model="anthropic--claude-4-sonnet")
    """
    
    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        **kwargs
    ):
        """
        Initialize AICoreLM with a gen-ai-hub-sdk client.
        
        Args:
            client: The SAP AI Core client from gen_ai_hub (e.g., Session().client())
            model_name: The model name (e.g., "anthropic--claude-4-sonnet")
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        """
        self.aicore_client = client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.additional_kwargs = kwargs
        
        # Initialize parent with a dummy model string since we're bypassing LiteLLM
        super().__init__(
            model=f"aicore/{model_name}",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def forward(self, prompt=None, messages=None, **kwargs):
        """
        Make a request to SAP AI Core.
        
        Args:
            prompt: Optional string prompt
            messages: Optional list of message dicts with 'role' and 'content'
            **kwargs: Additional arguments for the inference call
            
        Returns:
            A response object compatible with DSPy's expectations
        """
        # Convert prompt to messages format if needed
        if prompt and not messages:
            messages = [{"role": "user", "content": prompt}]
        
        if not messages:
            raise ValueError("Either prompt or messages must be provided")
        
        # Convert messages to SAP AI Core format
        aicore_messages = self._convert_messages_to_aicore_format(messages)
        
        # Merge temperature and max_tokens with kwargs
        inference_config = {
            "temperature": kwargs.get("temperature", self.temperature),
            "maxTokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        # Add topP if provided
        if "top_p" in kwargs:
            inference_config["topP"] = kwargs["top_p"]
        
        # Make the request to SAP AI Core
        try:
            response = self.aicore_client.converse(
                messages=aicore_messages,
                inferenceConfig=inference_config,
            )
            
            # Convert response to DSPy-compatible format
            return self._convert_aicore_response_to_dspy_format(response)
            
        except Exception as e:
            raise RuntimeError(f"Error calling SAP AI Core: {str(e)}") from e
    
    def _convert_messages_to_aicore_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert DSPy message format to SAP AI Core message format.
        
        DSPy format: [{"role": "user", "content": "text"}]
        AI Core format: [{"role": "user", "content": [{"text": "text"}]}]
        """
        aicore_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Convert content to AI Core format
            if isinstance(content, str):
                aicore_content = [{"text": content}]
            elif isinstance(content, list):
                # Already in the right format
                aicore_content = content
            else:
                aicore_content = [{"text": str(content)}]
            
            aicore_messages.append({
                "role": role,
                "content": aicore_content
            })
        
        return aicore_messages
    
    def _convert_aicore_response_to_dspy_format(self, response: Dict[str, Any]):
        """
        Convert SAP AI Core response to DSPy-compatible format.
        
        AI Core response structure (Bedrock-like):
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "response text"}]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }
        """
        # Create a mock LiteLLM response object
        class MockResponse:
            def __init__(self, content: str, usage: Dict[str, int]):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': content,
                        'role': 'assistant'
                    })()
                })()]
                self.usage = type('obj', (object,), usage)()
                self.cache_hit = False
        
        # Extract content from AI Core response
        try:
            output_message = response.get("output", {}).get("message", {})
            content_list = output_message.get("content", [])
            
            # Concatenate all text parts
            content = ""
            for item in content_list:
                if "text" in item:
                    content += item["text"]
            
            # Extract usage information
            usage_data = response.get("usage", {})
            usage = {
                "prompt_tokens": usage_data.get("inputTokens", 0),
                "completion_tokens": usage_data.get("outputTokens", 0),
                "total_tokens": usage_data.get("totalTokens", 0),
            }
            
            return MockResponse(content, usage)
            
        except Exception as e:
            raise RuntimeError(f"Error parsing SAP AI Core response: {str(e)}") from e


def create_aicore_lm(
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    **kwargs
) -> AICoreLM:
    """
    Convenience function to create an AICoreLM instance.
    
    This automatically initializes the SAP AI Core client using gen-ai-hub-sdk.
    
    Args:
        model_name: The model name (e.g., "anthropic--claude-4-sonnet")
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        **kwargs: Additional arguments
        
    Returns:
        AICoreLM instance
        
    Example:
        lm = create_aicore_lm(model_name="anthropic--claude-4-sonnet")
    """
    try:
        from gen_ai_hub.proxy.native.amazon.clients import Session
    except ImportError:
        raise ImportError(
            "gen-ai-hub-sdk is required for SAP AI Core support. "
            "Install it with: pip install 'kg-gen[aicore]'"
        )
    
    # Create the AI Core client
    client = Session().client(model_name=model_name)
    
    return AICoreLM(
        client=client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
