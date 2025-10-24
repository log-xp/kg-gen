"""
Test file demonstrating SAP AI Core integration with kg-gen.

To use this test, you need to:
1. Install the SAP AI Core SDK: pip install 'kg-gen[aicore]'
2. Configure your SAP AI Core credentials
3. Run the test

Note: This test is skipped by default if gen_ai_hub is not installed.
"""

import pytest
from src.kg_gen import KGGen

# Check if gen-ai-hub-sdk is available
try:
    from gen_ai_hub.proxy.native.amazon.clients import Session
    AICORE_AVAILABLE = True
except ImportError:
    AICORE_AVAILABLE = False


@pytest.mark.skipif(not AICORE_AVAILABLE, reason="gen-ai-hub-sdk not installed")
def test_aicore_basic():
    """Test basic knowledge graph generation with SAP AI Core."""
    
    # Create SAP AI Core client
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
    
    # Initialize KGGen with SAP AI Core
    kg = KGGen(
        aicore_client=bedrock,
        model="anthropic--claude-4-sonnet",
        temperature=0.5,
        max_tokens=512,
    )
    
    # Generate a simple graph
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter."
    
    graph = kg.generate(input_data=text)
    
    # Verify the graph contains expected entities
    expected_entities = {"Harry", "James Potter", "Lily Potter"}
    assert len(graph.entities) > 0
    print(f"Generated entities: {graph.entities}")
    print(f"Generated relations: {graph.relations}")
    print(f"Generated edges: {graph.edges}")


@pytest.mark.skipif(not AICORE_AVAILABLE, reason="gen-ai-hub-sdk not installed")
def test_aicore_conversation():
    """Test conversation-based knowledge graph generation with SAP AI Core."""
    
    # Create SAP AI Core client
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
    
    # Initialize KGGen with SAP AI Core
    kg = KGGen(
        aicore_client=bedrock,
        model="anthropic--claude-4-sonnet",
        temperature=0.5,
        max_tokens=512,
    )
    
    # Test with conversation format
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    
    graph = kg.generate(input_data=messages)
    
    # Verify the graph contains expected entities
    assert len(graph.entities) > 0
    print(f"Generated entities: {graph.entities}")
    print(f"Generated relations: {graph.relations}")


@pytest.mark.skipif(not AICORE_AVAILABLE, reason="gen-ai-hub-sdk not installed")
def test_aicore_with_clustering():
    """Test knowledge graph generation with clustering using SAP AI Core."""
    
    # Create SAP AI Core client
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
    
    # Initialize KGGen with SAP AI Core
    kg = KGGen(
        aicore_client=bedrock,
        model="anthropic--claude-4-sonnet",
        temperature=0.5,
        max_tokens=512,
    )
    
    # Generate two graphs
    text1 = "Linda is Joshua's mother. Ben is Josh's brother."
    text2 = "Andrew is Josh's father. Josh also goes by Joshua."
    
    graph1 = kg.generate(input_data=text1, context="Family relationships")
    graph2 = kg.generate(input_data=text2, context="Family relationships")
    
    # Aggregate and cluster
    combined_graph = kg.aggregate([graph1, graph2])
    clustered_graph = kg.cluster(combined_graph, context="Family relationships")
    
    # Verify clustering worked
    assert len(clustered_graph.entities) > 0
    print(f"Clustered entities: {clustered_graph.entities}")
    print(f"Entity clusters: {clustered_graph.entity_clusters}")


def test_aicore_example_from_problem_statement():
    """
    Example showing how to use SAP AI Core as described in the problem statement.
    
    This test demonstrates the exact usage pattern requested.
    """
    if not AICORE_AVAILABLE:
        pytest.skip("gen-ai-hub-sdk not installed")
    
    # This is the example from the problem statement
    from gen_ai_hub.proxy.native.amazon.clients import Session
    
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
    
    # Direct usage with the bedrock client (for reference)
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "text": "Describe the purpose of a 'hello world' program in one line."
                }
            ],
        }
    ]
    response = bedrock.converse(
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )
    print("Direct bedrock response:", response)
    
    # Now use with KGGen
    kg = KGGen(
        aicore_client=bedrock,
        model="anthropic--claude-4-sonnet",
        temperature=0.5,
        max_tokens=512,
    )
    
    text = "A hello world program is a simple program that prints 'Hello, World!' to demonstrate basic syntax."
    graph = kg.generate(input_data=text)
    
    print(f"KGGen entities: {graph.entities}")
    print(f"KGGen relations: {graph.relations}")
    
    assert len(graph.entities) > 0
