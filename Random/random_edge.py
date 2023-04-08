import random
import networkx as nx
import dgl

def Random(nx_G):
    G = nx_G.to_networkx()
    G = G.to_undirected()
    H = G.copy()
    p = 0.5
    q = 0.5
    nodes = list(G.nodes())
    adj_list = {u: set(G.neighbors(u)) for u in nodes}

    for i, u in enumerate(nodes):

        if i % 1000 == 0:
          
            print('the present step is:', i)

        adj_node = adj_list[u]

        if len(adj_node) != 0:
            nei_nei = set()
            for nei in adj_node:
                
                
                probability_1 = random.uniform(0, 1)

                if probability_1 > p and G.has_edge(u,nei):
                    G.remove_edge(u,nei)
                
                for nei_nei in set(H[nei]).difference(set(adj_node)):
                    if nei_nei != u:
                        probability_2 = random.uniform(0, 1)
                        if probability_2 > q and G.has_edge(u,nei_nei) == False:
                            G.add_edge(u,nei_nei)
                      
        G.add_edge(u, u)

    print('Edges operation is complete.')
    G = dgl.from_networkx(G)
    G.ndata['feat'] = nx_G.ndata['feat']

    print('Random is complete')
    return G
