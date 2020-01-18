import networkx as nx



print('Question 7 ================')

G=nx.Graph()
G.add_nodes_from([1,2,3,4,5])
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(4,5)
G.add_edge(5,1)
G.add_edge(5,3)

print(f'clustering: {nx.clustering(G)}, average clustering: {nx.average_clustering(G)}')
print(f'Overall clustering: {nx.transitivity(G)}')

print('Question 8 ================')
G=nx.Graph()
G.add_nodes_from([1,2,3,4,5,6,7])
G.add_edge(1,4)
G.add_edge(1,7)
G.add_edge(7,6)
G.add_edge(7,4)
G.add_edge(4,2)
G.add_edge(2,5)
G.add_edge(5,3)

print(f'degree centrality: {nx.degree_centrality(G)}')

print('Question 9 ================')

print(f'degree centrality of node 2: {nx.closeness_centrality(G)[2]}')

print('Question 10 ================')

print(f'betweenness centrality: {nx.betweenness_centrality(G)}')
