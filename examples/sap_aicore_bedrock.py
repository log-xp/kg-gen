"""
Example: Using SAP AI Core with Amazon Bedrock for Knowledge Graph Generation

This example demonstrates how to use SAP AI Core's Bedrock integration
with the kg-gen library for extracting knowledge graphs from text.

Requirements:
- SAP AI Core account with Bedrock access configured
- generative-ai-hub-sdk package installed
- Proper SAP AI Core credentials (handled automatically by gen_ai_hub SDK)
"""

from kg_gen import KGGen, SAPAICoreBedrockLM
import os

# Sample text from "The Name of the Wind" by Patrick Rothfuss
text = """
A Place for Demons
IT WAS FELLING NIGHT, and the usual crowd had gathered at the
Waystone Inn. Five wasn't much of a crowd, but five was as many as the
Waystone ever saw these days, times being what they were.
Old Cob was filling his role as storyteller and advice dispensary. The
men at the bar sipped their drinks and listened. In the back room a young
innkeeper stood out of sight behind the door, smiling as he listened to the
details of a familiar story.
"When he awoke, Taborlin the Great found himself locked in a high
tower. They had taken his sword and stripped him of his tools: key, coin,
and candle were all gone. But that weren't even the worst of it, you see…"
Cob paused for effect, "…cause the lamps on the wall were burning blue!"
Graham, Jake, and Shep nodded to themselves. The three friends had
grown up together, listening to Cob's stories and ignoring his advice.
Cob peered closely at the newer, more attentive member of his small
audience, the smith's prentice. "Do you know what that meant, boy?"
Everyone called the smith's prentice "boy" despite the fact that he was a
hand taller than anyone there. Small towns being what they are, he would
most likely remain "boy" until his beard filled out or he bloodied someone's
nose over the matter.
"""

# Initialize SAP AI Core Bedrock LM
# The model name format follows SAP AI Core naming: "provider--model-name"
sap_lm = SAPAICoreBedrockLM(
    model_name="anthropic--claude-4-sonnet",  # SAP AI Core model identifier
    max_tokens=4096,
    temperature=0.5,
    top_p=0.9
)

# Create KGGen instance with custom SAP AI Core LM
kg = KGGen(custom_lm=sap_lm)

# Generate knowledge graph
print("Generating knowledge graph using SAP AI Core Bedrock...")
graph = kg.generate(
    input_data=text,
    chunk_size=1000,
    cluster=True,
    context="Kingkiller Chronicles - Fantasy novel excerpt",
    output_folder="./examples/",
)

# Print statistics
print(f"\nKnowledge Graph Statistics:")
print(f"- Entities: {len(graph.entities)}")
print(f"- Relations: {len(graph.relations)}")
print(f"- Edges: {len(graph.edges)}")

if graph.entity_clusters:
    print(f"- Entity Clusters: {len(graph.entity_clusters)}")
if graph.edge_clusters:
    print(f"- Edge Clusters: {len(graph.edge_clusters)}")

# Visualize the graph
output_path = "./examples/sap-aicore-graph.html"
KGGen.visualize(graph, output_path, open_in_browser=False)
print(f"\nVisualization saved to: {output_path}")

# Example: Print some entities and relations
print("\nSample Entities:")
for entity in list(graph.entities)[:5]:
    print(f"  - {entity}")

print("\nSample Relations:")
for relation in list(graph.relations)[:5]:
    source, rel, target = relation
    print(f"  - {source} -> [{rel}] -> {target}")
