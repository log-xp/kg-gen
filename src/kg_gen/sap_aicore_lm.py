"""SAP AI Core Bedrock LM wrapper for DSPy integration."""

from typing import Any, Optional, List, Dict
import dspy
from gen_ai_hub.proxy.native.amazon.clients import Session


class SAPAICoreBedrockLM(dspy.LM):
    """
    DSPy-compatible Language Model wrapper for SAP AI Core with Amazon Bedrock.

    This class wraps the SAP AI Core Bedrock client to work seamlessly with
    the DSPy framework used throughout the kg-gen codebase.

    Example:
        ```python
        from kg_gen.sap_aicore_lm import SAPAICoreBedrockLM
        from kg_gen import KGGen

        # Create SAP AI Core LM
        lm = SAPAICoreBedrockLM(
            model_name="anthropic--claude-4-sonnet",
            max_tokens=4096,
            temperature=0.5
        )

        # Use with KGGen by passing as custom_lm
        kg = KGGen(custom_lm=lm)
        ```
    """

    def __init__(
        self,
        model_name: str = "anthropic--claude-4-sonnet",
        max_tokens: int = 4096,
        temperature: float = 0.5,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Initialize SAP AI Core Bedrock LM.

        Args:
            model_name: Name of the model in SAP AI Core (e.g., "anthropic--claude-4-sonnet")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent DSPy LM class
        super().__init__(model=model_name, **kwargs)

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize SAP AI Core Bedrock session and client
        self.session = Session()
        self.bedrock_client = self.session.client(model_name=model_name)

        # Store provider (extracted from model_name, e.g., "anthropic" from "anthropic--claude-4-sonnet")
        self.provider = model_name.split("--")[0] if "--" in model_name else "unknown"

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> List[str]:
        """
        Call the SAP AI Core Bedrock model.

        This method is called by DSPy when making LLM predictions.

        Args:
            prompt: Single string prompt (converted to messages format)
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            List of generated text responses
        """
        # Merge parameters: kwargs override instance defaults
        inference_config = {
            "maxTokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "topP": kwargs.get("top_p", self.top_p),
        }

        # Convert prompt to messages format if needed
        if messages is None:
            if prompt is None:
                raise ValueError("Either 'prompt' or 'messages' must be provided")
            messages = [{"role": "user", "content": [{"text": prompt}]}]
        else:
            # Convert DSPy message format to Bedrock format
            messages = self._convert_to_bedrock_format(messages)

        # Call Bedrock API
        response = self.bedrock_client.converse(
            messages=messages,
            inferenceConfig=inference_config,
        )

        # Extract generated text from response
        output_text = self._extract_response_text(response)

        # DSPy expects a list of completions
        return [output_text]

    def _convert_to_bedrock_format(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert DSPy message format to Bedrock converse API format.

        DSPy format: [{"role": "user", "content": "text"}]
        Bedrock format: [{"role": "user", "content": [{"text": "text"}]}]

        Args:
            messages: Messages in DSPy format

        Returns:
            Messages in Bedrock format
        """
        bedrock_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Convert content to Bedrock's nested format if needed
            if isinstance(content, str):
                bedrock_content = [{"text": content}]
            elif isinstance(content, list):
                # Already in correct format or needs conversion
                bedrock_content = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        bedrock_content.append(item)
                    elif isinstance(item, str):
                        bedrock_content.append({"text": item})
                    else:
                        bedrock_content.append({"text": str(item)})
            else:
                bedrock_content = [{"text": str(content)}]

            bedrock_messages.append({
                "role": role,
                "content": bedrock_content
            })

        return bedrock_messages

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """
        Extract generated text from Bedrock response.

        Bedrock response structure:
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "generated text"}]
                }
            },
            ...
        }

        Args:
            response: Response from Bedrock converse API

        Returns:
            Generated text string
        """
        try:
            # Navigate response structure
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])

            return " ".join(text_parts) if text_parts else ""

        except Exception as e:
            raise ValueError(f"Failed to extract response text: {e}\nResponse: {response}")

    def copy(self, **kwargs) -> "SAPAICoreBedrockLM":
        """
        Create a copy of this LM with updated parameters.

        Args:
            **kwargs: Parameters to update

        Returns:
            New SAPAICoreBedrockLM instance
        """
        params = {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        params.update(kwargs)
        return SAPAICoreBedrockLM(**params)
