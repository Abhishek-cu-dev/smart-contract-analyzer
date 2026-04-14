import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# ==========================================
# 1. STRUCTURAL BRANCH (GNN for Control Flow)
# ==========================================
class GNNBranch(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(GNNBranch, self).__init__()
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        # x: Node feature matrix, edge_index: Graph connectivity
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Pool graph into a single vector (Global Mean Pooling)
        x = global_mean_pool(x, batch)
        return x

# ==========================================
# 2. SEQUENTIAL BRANCH (LSTM for Opcodes)
# ==========================================
class LSTMBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1):
        super(LSTMBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # batch_first=True means input tensor is of shape (batch, seq, feature)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, opcodes):
        # opcodes shape: (batch_size, sequence_length)
        embedded = self.embedding(opcodes)
        
        # lstm_out contains all hidden states, (h_n, c_n) are the final states
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # We only need the final hidden state to represent the whole sequence
        final_state = h_n[-1] 
        return final_state

# ==========================================
# 3. SEMANTIC BRANCH (Attention for Source Code)
# ==========================================
class AttentionBranch(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super(AttentionBranch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Linear layer to condense the output
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens):
        # tokens shape: (batch_size, sequence_length)
        embedded = self.embedding(tokens)
        
        # Self-attention (query, key, and value are all the same)
        attn_output, attn_weights = self.attention(embedded, embedded, embedded)
        
        # Global average pooling over the sequence dimension to get a fixed-size vector
        pooled_output = torch.mean(attn_output, dim=1)
        out = F.relu(self.fc(pooled_output))
        
        return out, attn_weights # Returning weights for your "Explainability Map"

# ==========================================
# 4. FUSION LAYER & CLASSIFIER (The Core Engine)
# ==========================================
class HybridDeepLearningEngine(nn.Module):
    def __init__(self, gnn_out_dim=64, lstm_out_dim=64, attn_out_dim=64):
        super(HybridDeepLearningEngine, self).__init__()
        
        # Instantiate the 3 branches
        self.gnn_branch = GNNBranch(num_node_features=10, hidden_dim=gnn_out_dim)
        self.lstm_branch = LSTMBranch(vocab_size=500, embed_dim=32, hidden_dim=lstm_out_dim)
        self.attention_branch = AttentionBranch(vocab_size=1000, embed_dim=attn_out_dim, num_heads=4)
        
        # Calculate total dimension after concatenation
        fusion_dim = gnn_out_dim + lstm_out_dim + attn_out_dim
        
        # Deep Neural Network Classifier (DNN)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout prevents overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # Output layer (1 neuron for binary classification)
            nn.Sigmoid()      # Squeeze output between 0 and 1 (Confidence Score)
        )

    def forward(self, graph_x, edge_index, graph_batch, opcodes, tokens):
        # 1. Run GNN Branch
        gnn_out = self.gnn_branch(graph_x, edge_index, graph_batch)
        
        # 2. Run LSTM Branch
        lstm_out = self.lstm_branch(opcodes)
        
        # 3. Run Attention Branch
        attn_out, attn_weights = self.attention_branch(tokens)
        
        # 4. Concatenate Modalities (The Fusion Step)
        fused_vector = torch.cat((gnn_out, lstm_out, attn_out), dim=1)
        
        # 5. Classify Threat
        confidence_score = self.fusion_layer(fused_vector)
        
        return confidence_score, attn_weights
