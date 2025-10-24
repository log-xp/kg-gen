"""
Simple example matching the problem statement exactly.

This demonstrates using SAP AI Core with kg-gen in the simplest way possible.
"""

# Example from problem statement - direct SAP AI Core usage
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
print("Direct SAP AI Core response:")
print(response)
print()

# Now use the same client with kg-gen
from kg_gen import KGGen

kg = KGGen(
    aicore_client=bedrock,
    model="anthropic--claude-4-sonnet",
    temperature=0.5,
    max_tokens=512,
)

# Generate a knowledge graph
text = """
A hello world program is a simple program that prints 'Hello, World!' to demonstrate 
basic syntax. It's typically the first program written when learning a new programming 
language. The program serves as a test to verify that the development environment is 
properly configured.
"""

graph = kg.generate(input_data=text)

print("Knowledge Graph generated with kg-gen:")
print(f"Entities: {graph.entities}")
print(f"Relations: {graph.relations}")
print(f"Edges: {graph.edges}")
