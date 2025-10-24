"""
Test SAP AI Core integration with a mock client (no actual API calls).

This test validates that the AICoreLM wrapper correctly integrates with DSPy
without requiring actual SAP AI Core credentials.
"""

import pytest
from src.kg_gen.aicore_provider import AICoreLM


class MockAICoreClient:
    """Mock SAP AI Core client for testing."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def converse(self, messages, inferenceConfig):
        """Mock converse method that returns a fake response."""
        # Return a mock response in the expected format
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "This is a mock response with entities: Alice, Bob. Alice knows Bob."}
                    ]
                }
            },
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }


def test_aicore_lm_initialization():
    """Test that AICoreLM can be initialized with a mock client."""
    mock_client = MockAICoreClient("test-model")
    lm = AICoreLM(
        client=mock_client,
        model_name="test-model",
        temperature=0.5,
        max_tokens=100
    )
    
    assert lm.model_name == "test-model"
    assert lm.temperature == 0.5
    assert lm.max_tokens == 100
    print("✓ AICoreLM initialization successful")


def test_aicore_lm_message_conversion():
    """Test that AICoreLM correctly converts message formats."""
    mock_client = MockAICoreClient("test-model")
    lm = AICoreLM(
        client=mock_client,
        model_name="test-model",
    )
    
    # Test DSPy format to AI Core format conversion
    dspy_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]
    
    aicore_messages = lm._convert_messages_to_aicore_format(dspy_messages)
    
    assert len(aicore_messages) == 2
    assert aicore_messages[0]["role"] == "user"
    assert aicore_messages[0]["content"] == [{"text": "Hello"}]
    assert aicore_messages[1]["role"] == "assistant"
    assert aicore_messages[1]["content"] == [{"text": "Hi there"}]
    print("✓ Message conversion successful")


def test_aicore_lm_forward():
    """Test that AICoreLM forward method works with mock client."""
    mock_client = MockAICoreClient("test-model")
    lm = AICoreLM(
        client=mock_client,
        model_name="test-model",
    )
    
    # Test with simple prompt
    response = lm.forward(prompt="Test prompt")
    
    assert hasattr(response, 'choices')
    assert len(response.choices) > 0
    assert hasattr(response.choices[0], 'message')
    assert hasattr(response.choices[0].message, 'content')
    assert "mock response" in response.choices[0].message.content.lower()
    print("✓ Forward method successful")


def test_aicore_lm_with_messages():
    """Test that AICoreLM works with messages format."""
    mock_client = MockAICoreClient("test-model")
    lm = AICoreLM(
        client=mock_client,
        model_name="test-model",
    )
    
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    response = lm.forward(messages=messages)
    
    assert hasattr(response, 'choices')
    assert hasattr(response, 'usage')
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 30
    print("✓ Messages format successful")


def test_kggen_with_mock_aicore():
    """Test that KGGen works with mock AI Core client."""
    from src.kg_gen import KGGen
    
    mock_client = MockAICoreClient("test-model")
    
    # Initialize KGGen with mock AI Core client
    kg = KGGen(
        aicore_client=mock_client,
        model="test-model",
        temperature=0.5,
        max_tokens=100
    )
    
    # Verify the LM is an AICoreLM instance
    assert hasattr(kg, 'lm')
    assert isinstance(kg.lm, AICoreLM)
    assert kg.aicore_client == mock_client
    print("✓ KGGen with mock AI Core client successful")


def test_aicore_response_parsing():
    """Test that AI Core responses are correctly parsed."""
    mock_client = MockAICoreClient("test-model")
    lm = AICoreLM(
        client=mock_client,
        model_name="test-model",
    )
    
    # Mock AI Core response
    aicore_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Part 1 "},
                    {"text": "Part 2"}
                ]
            }
        },
        "usage": {
            "inputTokens": 5,
            "outputTokens": 10,
            "totalTokens": 15
        }
    }
    
    parsed = lm._convert_aicore_response_to_dspy_format(aicore_response)
    
    assert hasattr(parsed, 'choices')
    assert parsed.choices[0].message.content == "Part 1 Part 2"
    assert parsed.usage.prompt_tokens == 5
    assert parsed.usage.completion_tokens == 10
    assert parsed.usage.total_tokens == 15
    print("✓ Response parsing successful")


if __name__ == "__main__":
    print("Running AI Core mock tests...\n")
    test_aicore_lm_initialization()
    test_aicore_lm_message_conversion()
    test_aicore_lm_forward()
    test_aicore_lm_with_messages()
    test_kggen_with_mock_aicore()
    test_aicore_response_parsing()
    print("\n✅ All mock tests passed!")
