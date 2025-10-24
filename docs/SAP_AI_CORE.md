# Using kg-gen with SAP AI Core

This guide explains how to use kg-gen with SAP AI Core's LLM proxy service instead of directly calling OpenAI or other providers.

## Overview

SAP AI Core provides enterprise-grade LLM services with models from various providers (OpenAI, Anthropic, etc.) through a unified API. The `kg-gen` library now supports SAP AI Core through the `generative-ai-hub-sdk`.

## Installation

Install kg-gen with SAP AI Core support:

```bash
pip install 'kg-gen[aicore]'
```

This will install:
- `kg-gen` - The knowledge graph generation library
- `generative-ai-hub-sdk` - SAP AI Core SDK for Python

## Prerequisites

Before using kg-gen with SAP AI Core, you need:

1. Access to an SAP AI Core instance
2. Proper authentication credentials configured
3. A deployed model in SAP AI Core (e.g., "anthropic--claude-4-sonnet")

For detailed SAP AI Core setup instructions, refer to the [official SAP AI Core documentation](https://help.sap.com/docs/ai-core).

## Quick Start

### Basic Usage

```python
from gen_ai_hub.proxy.native.amazon.clients import Session
from kg_gen import KGGen

# Create SAP AI Core client
bedrock = Session().client(model_name="anthropic--claude-4-sonnet")

# Initialize KGGen with SAP AI Core
kg = KGGen(
    aicore_client=bedrock,
    model="anthropic--claude-4-sonnet",
    temperature=0.5,
    max_tokens=512,
)

# Generate a knowledge graph
text = "Your text here..."
graph = kg.generate(input_data=text)

print(f"Entities: {graph.entities}")
print(f"Relations: {graph.relations}")
```

### Using Conversation Format

```python
from gen_ai_hub.proxy.native.amazon.clients import Session
from kg_gen import KGGen

# Create SAP AI Core client
bedrock = Session().client(model_name="anthropic--claude-4-sonnet")

# Initialize KGGen
kg = KGGen(
    aicore_client=bedrock,
    model="anthropic--claude-4-sonnet",
)

# Process a conversation
messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI..."},
]

graph = kg.generate(input_data=messages)
```

### With Clustering

```python
from gen_ai_hub.proxy.native.amazon.clients import Session
from kg_gen import KGGen

# Create SAP AI Core client
bedrock = Session().client(model_name="anthropic--claude-4-sonnet")

# Initialize KGGen
kg = KGGen(
    aicore_client=bedrock,
    model="anthropic--claude-4-sonnet",
)

# Generate multiple graphs
text1 = "Linda is Joshua's mother. Ben is Josh's brother."
text2 = "Josh also goes by Joshua."

graph1 = kg.generate(input_data=text1, context="Family relationships")
graph2 = kg.generate(input_data=text2, context="Family relationships")

# Aggregate and cluster
combined_graph = kg.aggregate([graph1, graph2])
clustered_graph = kg.cluster(combined_graph, context="Family relationships")

print(f"Entity clusters: {clustered_graph.entity_clusters}")
```

## Supported Models

Any model deployed in your SAP AI Core instance can be used. Common models include:

- `anthropic--claude-4-sonnet`
- `anthropic--claude-3-5-sonnet`
- `gpt-4o` (if configured)
- `gpt-4-turbo` (if configured)

The exact model names depend on your SAP AI Core deployment configuration.

## Configuration Parameters

When initializing `KGGen` with SAP AI Core:

- `aicore_client` (required): The SAP AI Core client from `gen_ai_hub`
- `model` (required): The model name as configured in SAP AI Core
- `temperature` (optional): Sampling temperature (default: 0.0)
- `max_tokens` (optional): Maximum tokens to generate (default: 4000)

## Advanced Usage

### Using Different Models for Different Operations

```python
from gen_ai_hub.proxy.native.amazon.clients import Session
from kg_gen import KGGen

# Create clients for different models
claude = Session().client(model_name="anthropic--claude-4-sonnet")
gpt = Session().client(model_name="gpt-4o")

# Use Claude for generation
kg = KGGen(aicore_client=claude, model="anthropic--claude-4-sonnet")
graph = kg.generate(input_data=text)

# Use GPT for clustering
clustered = kg.cluster(
    graph, 
    context="...",
    aicore_client=gpt
)
```

### Direct API Usage

You can also use the SAP AI Core client directly:

```python
from gen_ai_hub.proxy.native.amazon.clients import Session

bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
conversation = [
    {
        "role": "user",
        "content": [{"text": "Your question here"}],
    }
]
response = bedrock.converse(
    messages=conversation,
    inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
)
```

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'gen_ai_hub'`:

```bash
pip install 'kg-gen[aicore]'
```

### Authentication Error

If you get authentication errors, ensure your SAP AI Core credentials are properly configured. Refer to the [SAP AI Core documentation](https://help.sap.com/docs/ai-core) for credential setup.

### Model Not Found

If you get a "model not found" error, verify:
1. The model is deployed in your SAP AI Core instance
2. The model name matches the deployment configuration exactly
3. You have access to the model

## Examples

See the following example files:
- `examples/aicore_simple.py` - Simple example matching the problem statement
- `examples/aicore_example.py` - Comprehensive examples with various use cases
- `tests/test_aicore.py` - Integration tests (require credentials)
- `tests/test_aicore_mock.py` - Unit tests with mock client (no credentials needed)

## Benefits of Using SAP AI Core

1. **Enterprise Support**: Professional support from SAP
2. **Security**: Enterprise-grade security and compliance
3. **Model Flexibility**: Access to multiple LLM providers through one API
4. **Cost Management**: Centralized billing and cost tracking
5. **Governance**: Centralized model governance and access control

## Additional Resources

- [SAP AI Core Documentation](https://help.sap.com/docs/ai-core)
- [generative-ai-hub-sdk on PyPI](https://pypi.org/project/generative-ai-hub-sdk/)
- [kg-gen Documentation](https://github.com/stair-lab/kg-gen)
