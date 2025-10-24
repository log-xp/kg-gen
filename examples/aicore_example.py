"""
Example: Using kg-gen with SAP AI Core

This example demonstrates how to use kg-gen with SAP AI Core's LLM proxy service
instead of directly calling OpenAI or other providers.

Prerequisites:
1. Install kg-gen with AI Core support:
   pip install 'kg-gen[aicore]'

2. Configure SAP AI Core credentials according to gen-ai-hub-sdk documentation

3. Have access to an AI Core deployment with a supported model
"""

from gen_ai_hub.proxy.native.amazon.clients import Session
from kg_gen import KGGen


def example_basic_usage():
    """Basic usage with SAP AI Core."""
    
    print("=== Basic Usage with SAP AI Core ===\n")
    
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
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together."
    
    graph = kg.generate(input_data=text)
    
    print(f"Entities: {graph.entities}")
    print(f"Relations: {graph.relations}")
    print(f"Edges: {graph.edges}")


def example_conversation_format():
    """Using conversation format with SAP AI Core."""
    
    print("\n=== Conversation Format with SAP AI Core ===\n")
    
    # Create SAP AI Core client
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
    
    # Initialize KGGen
    kg = KGGen(
        aicore_client=bedrock,
        model="anthropic--claude-4-sonnet",
        temperature=0.5,
        max_tokens=512,
    )
    
    # Process a conversation
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."},
        {"role": "user", "content": "What about deep learning?"},
        {"role": "assistant", "content": "Deep learning is a type of machine learning that uses neural networks with multiple layers."},
    ]
    
    graph = kg.generate(input_data=messages)
    
    print(f"Entities: {graph.entities}")
    print(f"Relations: {graph.relations}")


def example_with_clustering():
    """Using clustering with SAP AI Core."""
    
    print("\n=== Clustering with SAP AI Core ===\n")
    
    # Create SAP AI Core client
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
    
    # Initialize KGGen
    kg = KGGen(
        aicore_client=bedrock,
        model="anthropic--claude-4-sonnet",
        temperature=0.5,
        max_tokens=512,
    )
    
    # Generate multiple graphs
    text1 = "Linda is Joshua's mother. Ben is Josh's brother. Andrew is Josh's father."
    text2 = "Judy is Andrew's sister. Josh is Judy's nephew. Josh also goes by Joshua."
    
    graph1 = kg.generate(input_data=text1, context="Family relationships")
    graph2 = kg.generate(input_data=text2, context="Family relationships")
    
    # Aggregate and cluster
    combined_graph = kg.aggregate([graph1, graph2])
    clustered_graph = kg.cluster(combined_graph, context="Family relationships")
    
    print(f"Entities: {clustered_graph.entities}")
    print(f"Entity Clusters: {clustered_graph.entity_clusters}")
    print(f"Relations: {clustered_graph.relations}")


def example_direct_aicore_client():
    """
    Example showing direct usage of SAP AI Core client
    (as shown in the problem statement).
    """
    
    print("\n=== Direct SAP AI Core Client Usage ===\n")
    
    from gen_ai_hub.proxy.native.amazon.clients import Session
    
    bedrock = Session().client(model_name="anthropic--claude-4-sonnet")
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
    print(response)


if __name__ == "__main__":
    # Run examples
    try:
        example_basic_usage()
        example_conversation_format()
        example_with_clustering()
        example_direct_aicore_client()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure to install kg-gen with AI Core support:")
        print("  pip install 'kg-gen[aicore]'")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your SAP AI Core credentials are properly configured.")
