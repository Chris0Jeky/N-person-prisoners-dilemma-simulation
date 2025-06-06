import networkx as nx

# Test the exact case that's failing
print("Testing watts_strogatz_graph(4, 3, 0.0):")
G = nx.watts_strogatz_graph(4, 3, 0.0)
print(f"Nodes: {list(G.nodes())}")
print(f"Edges: {list(G.edges())}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Degree sequence: {[G.degree(n) for n in G.nodes()]}")

# Try with even k
print("\nTesting watts_strogatz_graph(4, 2, 0.0):")
G2 = nx.watts_strogatz_graph(4, 2, 0.0)
print(f"Edges: {list(G2.edges())}")
print(f"Number of edges: {G2.number_of_edges()}")

# What about k=4 with n=4? 
print("\nTesting watts_strogatz_graph(4, 4, 0.0):")
try:
    G3 = nx.watts_strogatz_graph(4, 4, 0.0)
    print(f"Edges: {list(G3.edges())}")
    print(f"Number of edges: {G3.number_of_edges()}")
except Exception as e:
    print(f"Error: {e}")
