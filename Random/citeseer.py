import dgl
import torch
import numpy as np
import kcore
import random_edge
import torch.nn.functional as F
from dgl.data import CiteseerGraphDataset
from dgl.nn import GraphConv
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('CUDA version\t:', torch.version.cuda)

# Load the AmazonCoBuy dataset
data = CiteseerGraphDataset()
g = data[0]
num_class = data.num_classes
# get node feature
feat = g.ndata['feat']
print(feat.shape[1])
# get data split
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
# get labels
labels = g.ndata['label']

G = random_edge.Random(g)


# Define a GNN model
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, num_classes)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

# Initialize the GNN model and optimizer
model = GCN(feat.shape[1], 64, data.num_classes)

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
        val_f1 = f1_score(labels[val_mask].numpy(), logits[val_mask].argmax(dim=1).numpy(), average='micro')

    # Print the training and validation metrics
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

# Evaluate the GNN model on the test set
model.eval()
with torch.no_grad():
    logits = model(G, feat)
    test_loss = F.cross_entropy(logits[test_mask], labels[test_mask])
    test_acc = (logits[test_mask].argmax(dim=1) == labels[test_mask]).float().mean()
    test_f1 = f1_score(labels[test_mask].numpy(), logits[test_mask].argmax(dim=1).numpy(), average='micro')
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")