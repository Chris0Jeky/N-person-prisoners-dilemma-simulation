import networkx as nx

# Test the behavior of watts_strogatz_graph with n=4, k=3, p=0.0
print("Testing nx.watts_strogatz_graph(4, 3, 0.0):")
graph = nx.watts_strogatz_graph(4, 3, 0.0)
print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")
print(f"Edges: {list(graph.edges())}")
print(f"Is complete graph? {nx.is_isomorphic(graph, nx.complete_graph(4))}")

print("\nTesting nx.watts_strogatz_graph(4, 2, 0.0):")
graph2 = nx.watts_strogatz_graph(4, 2, 0.0)
print(f"Number of nodes: {graph2.number_of_nodes()}")
print(f"Number of edges: {graph2.number_of_edges()}")
print(f"Edges: {list(graph2.edges())}")

# Check if it's a cycle vs complete graph
print("\nFor comparison, complete graph on 4 nodes:")
K4 = nx.complete_graph(4)
print(f"K4 edges: {K4.number_of_edges()}")
print(f"K4 edge list: {list(K4.edges())}")
