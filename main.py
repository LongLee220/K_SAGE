import numpy as np
import networkx as nx
import dgl
import copy
import torch as th
import scipy.sparse as ss
import copy 
import dgl
from dgl.data import AmazonCoBuyComputerDataset
import torch
import torch.nn.functional as F
import dgl.nn as dglnn
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import f1_score


def Kcore_community(nx_G):
    G = nx_G.to_networkx()
    G = G.to_undirected()
    Degree = {}
    Nodes = list(G.nodes())

    Nodes_u = copy.deepcopy(Nodes)
    #print(G[2201])
    Degree = {}
    for node in G.nodes():
        Degree[node] = len(list(G[node]))
        
    Degree_u = copy.deepcopy(Degree)
    rows = []
    cols = []
    vals = []
    for edge in G.edges():
        rows.append(edge[0])
        rows.append(edge[1])
        cols.append(edge[1])
        cols.append(edge[0])
        vals.append(int(1))
        vals.append(int(1))
                
    n = max(rows)+1
    A = ss.coo_matrix((vals, (rows,cols)),shape = (n, n))
    
    #E = len(G.edges())
    #print('The step is complete.',len(G.edges()))
    ##K-core

    def K_core(G,A):
        a = A.shape
        n = a[0]
        Node = set(np.arange(n))
        kcore = {}
        for node in set(G.nodes()).difference(Node):
            kcore[node] = 0
        b = np.ones((n,1))
        
        D = A.dot(b)
        # num_max = int(max(D))
        mun_min = int(min(D))
        
        # for k in range(mun_min,num_max):
        k = mun_min
        while len(Node) != 0:
            
            Flag = True
            while Flag == True:
                Flag = False
                
                Removed_list = []
                c = np.zeros((n,1))
                for node in Node:
                    if D[node] <= k and b[node] == 1:
                        
                        Removed_list.append(node)
                        
                        c[node] = 1
                        b[node] = 0
                        kcore[node] = k
                        
                        
                if len(Removed_list) !=0:
                    D = D - A.dot(c)
                    Node = set(Node).difference(Removed_list)
                    
                    Flag = True
                    
                else:
                    k = k+1
                    Flag = False
    
        return(kcore)

    kcore = K_core(G, A)
    print('K_core is complete.')
    
    def GIF_nodes(G, kcore, Degree_u):
        IF = {}
        for node in G.nodes():
            IF[node] = Degree_u[node] * kcore[node]
            
        Max_IF = max(IF.values())

        GIF = {}
        for node in G.nodes():
            GIF[node] = IF[node] / Max_IF
            
        return GIF
    
    GIF = GIF_nodes(G, kcore, Degree_u)
    
    print('GIF is complete.')
    
    
    def Select_nodes(G,Node_u,GIF):
        
        #Attr = np.zeros(A.shape)
        #P = np.zeros(A.shape)
        Max_strong = {}
        for ide1 in range(len(Node_u)):
            node1 = Node_u[ide1]
            
            adj_node = set(G[node1])
            
            adj_node_u = copy.deepcopy(adj_node)
            
            if len(adj_node) == 0:
                Max_strong[node1] = node1
            else:

                #2阶邻居
                Adj_path = {}
                Adj_path[node1] = 0
                #2阶邻居
                Nei_Nei = set()
                for nei in adj_node_u:
                    
                    Adj_path[nei] = 1
                    
                    for nei_nei in G[nei]:
                        if nei_nei not in adj_node:
                            Adj_path[nei_nei] = 2
                            Nei_Nei.add(nei_nei)
                        
                adj_node = adj_node|Nei_Nei
                
                Attr = {}
                P = {}
                adj_node = list(Adj_path.keys())
                for node2 in adj_node:
                    
                    if node2 != node1:
                        Triangle = set(G[node1]).intersection(set(G[node2]))
                        P[node2] = (len(Triangle)+1)/Degree_u[node1]
                        Attr[node2] = P[node2] *(GIF[node1]*GIF[node2])/(Adj_path[node2]**2)
                    else:
                        Attr[node2] = 0

                if ide1 % 10000 == 0:
                    print('the present step is:',ide1)    
                #if len(Attr) == 0:
                #    print(node1,adj_node)
                max_val = max(Attr.values())
                maxattr = {}#最大吸引
                minsl = {}#最小距离
                
                for nei in list(Attr.keys()):
                    if Attr[nei] == max_val:
                        maxattr[nei] = max_val
                        minsl[nei] = Adj_path[nei]
                if len(maxattr) == 1:
                    Max_strong[node1] = list(maxattr.keys())[0]
                else:
                    if len(set(minsl.values())) == 1:
                        Max_strong[node1] = list(maxattr.keys())[0]
                    else:
                        Max_strong[node1] = min(minsl,key=minsl.get)
                    
        return Max_strong
            
    
    Max_strong = Select_nodes(G, Nodes_u,GIF)
    
    print('Max_strong is complete')
    
    def Community(Max_strong, Nodes_u):
        Remove = set()
        Com = []
    
        for node1 in Nodes_u:
            if node1 in Remove:
                continue
    
            can_com = {node1, Max_strong[node1]}
            map_c = {Max_strong[node1]}
            Remove.add(node1)
            
            adj_node = set(G[node1])
            adj_node_u = copy.deepcopy(adj_node)
            #2阶邻居
            Nei_Nei = set()
            for nei in adj_node_u:
                
                for nei_nei in G[nei]:
                    if nei_nei not in adj_node:
                        Nei_Nei.add(nei_nei)
                    
            adj_node = adj_node|Nei_Nei
            Flag = True
            while Flag:
                Flag = False
                for node2 in adj_node:
                    if node2 in Remove:
                        continue
    
                    if Max_strong[node2] in can_com or node2 in map_c:
                        can_com.add(node2)
                        map_c.add(Max_strong[node2])
                        Remove.add(node2)
                        Flag = True
    
            Com.append(can_com)
    
        return Com

    Com = Community(Max_strong, Nodes_u)
    
    print('Com is complete.')
    
    def Edges_optration(G, Com, kcore):
        #Edges operations
        Nodes_com_ide = {}
        t = 1
        for ide in range(len(Com)):
            com_m = Com[ide]
            for node1 in com_m:
                
                Nodes_com_ide[node1] = ide
                for node2 in com_m:
                    if kcore[node1] == kcore[node2] and G.has_edge(node1,node2) == False:
                        G.add_edge(node1,node2)
                    else:
                        if G.has_edge(node1,node2) == True:
                            G.remove_edge(node1,node2)
        # Nodes = Nodes_u
        #        if G.has_edge(node1,node1) == False:
        #           G.add_edge(node1,node1)
        H = copy.deepcopy(G)  
        for edge in H.edges():
            node1 = edge[0]
            node2 = edge[1]
            if Nodes_com_ide[node1] != Nodes_com_ide[node2]:
                continue

            if kcore[node1] != kcore[node2]:
                G.remove_edge(node1,node2)
        return G

    G = Edges_optration(G, Com, kcore)
    print('Edges_optration is complete.')
    G = dgl.from_networkx(G)
    G.ndata['feat'] = nx_G.ndata['feat']
    return G


# Construct a DGLGraph
data = PubmedGraphDataset()
g = data[0]

##Homogeneity
# Define the labels of the nodes
labels = g.ndata['label']
size = (g.edges()[1].shape)
M = int(size[0])
print('the edges of g:',M)

# Compute the homogeneity of the graph
homogeneity = 0
for src, dst in zip(*g.edges()):
    if labels[src] == labels[dst]:
        homogeneity += 1

print('Homogeneity:', homogeneity/M)


G= Kcore_community(g)

size_G = (G.edges()[1].shape)
M = int(size_G[0])
print('the edges of G:',M)
homogeneity_G = 0
for src, dst in zip(*G.edges()):
    #rint(src,dst)
    if labels[src] == labels[dst]:
        homogeneity_G += 1

print('Homogeneity of G:', homogeneity_G/M)


num_class = data.num_classes
# get node feature
feat = g.ndata['feat']
print(feat.shape[1])

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

N = g.number_of_nodes()
train_num = int(N * train_ratio)
val_num = int(N * (train_ratio + val_ratio))

idx = np.arange(N)
np.random.shuffle(idx)

train_idx = idx[:train_num]
val_idx = idx[train_num:val_num]
test_idx = idx[val_num:]

train_mask = torch.tensor(train_idx)
val_mask = torch.tensor(val_idx)
test_mask = torch.tensor(test_idx)



# Define a GNN model
class SAGE(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


# Initialize the GNN model and optimizer
model = SAGE(in_feats=feat.shape[1], hid_feats=64, out_feats=data.num_classes)
optimizer = Adam(model.parameters())

# Train the GNN model
for epoch in range(200):
    model.train()
    logits = model(G, feat)
    train_loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    train_acc = (logits[train_mask].argmax(dim=1) == labels[train_mask]).float().mean()
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Evaluate the GNN model on the validation set
    model.eval()
    with torch.no_grad():
        logits = model(G, feat)
        val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
        val_acc = (logits[val_mask].argmax(dim=1) == labels[val_mask]).float().mean()
    # Print the training and validation metrics
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate the GNN model on the test set
model.eval()
with torch.no_grad():
    logits = model(G, feat)
    test_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
    test_acc = (logits[test_mask].argmax(dim=1) == labels[test_mask]).float().mean()
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")