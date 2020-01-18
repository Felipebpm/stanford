import networkx as nx
import operator
import sys

network = nx.read_pajek('../Data/imports_manufactures/imports_manufactures.net')

print('\nQuestion 13 ================')
country_centrality = nx.closeness_centrality(network)
country_centrality_inverse = [(value, key) for key, value in country_centrality.items()]

# For test purposes:
# print(f'closeness centrality: {country_centrality}')

print(f'Best country closeness centrality: {max(country_centrality_inverse)[1]}')

print('\nQuestion 14 ================')
country_betweenness_centrality = nx.betweenness_centrality(network)
country_betweenness_centrality_inverse = [(value, key) for key, value in country_betweenness_centrality.items()]

print(f'Best country betweenness centrality: {max(country_betweenness_centrality_inverse)[1]}')

print('\nQuestion 15 ================')

node_count = 400
probabilities = [0.0025, 0.0125, 0.025, 0.05]

for p in probabilities:
    graph = nx.erdos_renyi_graph(node_count, p)
    print(f'Erdos Renyi graph with {node_count} nodes and p: {p} is connected: {nx.is_connected(graph)}')
