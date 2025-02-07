import matplotlib.pyplot as plt
import networkx as nx

# Create the directed graph
G = nx.DiGraph()

# Add nodes for the main components
G.add_node("Social Flocking Model", layer=0)
G.add_node("Bot Agents", layer=1)
G.add_node("Platform AI", layer=1)
G.add_node("User Agents", layer=1)

# Add sub-components for Bot Agents
G.add_node("AdBotAgent", layer=2)
G.add_node("ShillBotAgent", layer=2)
G.add_node("OriginalPostAgent", layer=2)

# Add sub-components for Platform AI
G.add_node("Detection Module", layer=2)
G.add_node("Adaptive Learning", layer=2)

# Add sub-components for User Agents
G.add_node("Credibility Assessment", layer=2)
G.add_node("Decision-Making Model", layer=2)

# Connect the main model to its components
G.add_edges_from([
    ("Social Flocking Model", "Bot Agents"),
    ("Social Flocking Model", "Platform AI"),
    ("Social Flocking Model", "User Agents"),
    
    ("Bot Agents", "AdBotAgent"),
    ("Bot Agents", "ShillBotAgent"),
    ("Bot Agents", "OriginalPostAgent"),
    
    ("Platform AI", "Detection Module"),
    ("Platform AI", "Adaptive Learning"),
    
    ("User Agents", "Credibility Assessment"),
    ("User Agents", "Decision-Making Model"),
])

# Positioning the nodes
pos = nx.multipartite_layout(G, subset_key="layer")

# Drawing the graph
plt.figure(figsize=(14, 10))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=4500, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Social Media Collaborative Manipulation Simulation Framework", fontsize=18)
plt.show()
