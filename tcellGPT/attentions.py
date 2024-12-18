import torch
import torch.nn as nn
class SparseRandomAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_sparse_connections):
        super(SparseRandomAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.num_sparse_connections = num_sparse_connections

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by the number of heads."

        # Linear projections for queries and keys (we skip values since output is not needed)
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # init all parameters
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)


    def forward(self, embedding_matrix):
        # embedding_matrix: (num_nodes, embedding_dim)
        num_nodes = embedding_matrix.size(0)

        # Project queries and keys
        queries = self.query_proj(embedding_matrix)  # (num_nodes, embedding_dim)
        keys = self.key_proj(embedding_matrix)       # (num_nodes, embedding_dim)

        # Split into multiple heads and reshape for multi-head attention
        queries = queries.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_nodes, head_dim)
        keys = keys.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)        # (num_heads, num_nodes, head_dim)

        # Initialize sparse attention weight matrix with zeros
        sparse_attn_weights = torch.zeros(self.num_heads, num_nodes, num_nodes, device=embedding_matrix.device)

        # Generate random sparse connections
        # Randomly sample node indices for each head
        sparse_indices = torch.randint(0, num_nodes, (self.num_heads, num_nodes, self.num_sparse_connections), device=embedding_matrix.device)

        # Loop through each head to compute sparse attention weights
        for i in range(self.num_heads):
            # Select sparse keys for this head
            selected_keys = keys[i][sparse_indices[i]]  # (num_nodes, num_sparse_connections, head_dim)

            # Compute attention scores for selected pairs
            query = queries[i].unsqueeze(1)  # (num_nodes, 1, head_dim)
            attn_scores = torch.bmm(query, selected_keys.transpose(1, 2)).squeeze(1)  # (num_nodes, num_sparse_connections)
            attn_scores = attn_scores / (self.head_dim ** 0.5)  # Scaled dot-product

            # Apply softmax to get sparse attention weights
            #attn_weights = F.softmax(attn_scores, dim=-1)  # (num_nodes, num_sparse_connections)

            # Place sparse attention weights in the larger sparse_attn_weights matrix
            sparse_attn_weights[i].scatter_(1, sparse_indices[i], attn_scores)

        return sparse_attn_weights

class SparseAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_sparse_connections):
        super(SparseAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.num_sparse_connections = num_sparse_connections

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by the number of heads."

        # Linear projections for queries and keys
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)

        # Initialize parameters uniformly and multiply by very small weight
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def forward(self, embedding_matrix):
        # embedding_matrix: (num_nodes, embedding_dim)
        num_nodes = embedding_matrix.size(0)

        # Project queries, keys, and values
        queries = self.query_proj(embedding_matrix)  # (num_nodes, embedding_dim)
        keys = self.key_proj(embedding_matrix)       # (num_nodes, embedding_dim)

        # Split into multiple heads and reshape for multi-head attention
        queries = queries.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_nodes, head_dim)
        keys = keys.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)        # (num_heads, num_nodes, head_dim)

        # Initialize sparse attention weight matrix with zeros
        sparse_attn_weights = torch.zeros(self.num_heads, num_nodes, num_nodes, device=embedding_matrix.device)

        # Generate sparse connections based on similarity scores
        for i in range(self.num_heads):
            # Compute similarity scores between queries and all keys
            query = queries[i].unsqueeze(0)  # (num_nodes, 1, head_dim)
            similarity_scores = torch.bmm(query, keys[i].unsqueeze(0).transpose(1, 2)).squeeze(0)  # (num_nodes, num_nodes)
            similarity_scores = similarity_scores / (self.head_dim ** 0.5)  # Scale scores

            # For each node, select indices of the top num_sparse_connections most similar nodes
            top_k_indices = torch.topk(similarity_scores, self.num_sparse_connections, dim=-1).indices  # (num_nodes, num_sparse_connections)

            # Compute attention scores for these top connections
            selected_keys = keys[i][top_k_indices]  # (num_nodes, num_sparse_connections, head_dim)
            attn_scores = torch.bmm(query.transpose(0,1), selected_keys.transpose(1, 2)).squeeze(1)  # (num_nodes, num_sparse_connections)
            attn_scores = attn_scores / (self.head_dim ** 0.5)

            # Apply softmax to get sparse attention weights
            #attn_weights = F.softmax(attn_scores, dim=-1)  # (num_nodes, num_sparse_connections)

            # Place sparse attention weights in the larger sparse_attn_weights matrix
            sparse_attn_weights[i].scatter_(1, top_k_indices, attn_scores)

        return sparse_attn_weights


class MultiheadNodeAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_sparse_connections):
        super(MultiheadNodeAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        assert self.head_dim * num_heads == embedding_dim, "Embedding dimension must be divisible by the number of heads."

        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)


    def forward(self, embedding_matrix):
        # embedding_matrix: (num_nodes, embedding_dim)
        num_nodes = embedding_matrix.size(0)

        # Project queries, keys, and values
        queries = self.query_proj(embedding_matrix)  # (num_nodes, embedding_dim)
        keys = self.key_proj(embedding_matrix)      # (num_nodes, embedding_dim)

        # Split into multiple heads and reshape for multi-head attention
        queries = queries.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_nodes, head_dim)
        keys = keys.view(num_nodes, self.num_heads, self.head_dim).transpose(0, 1)        # (num_heads, num_nodes, head_dim)

        # Scaled dot-product attention
        # Compute attention scores
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.head_dim ** 0.5)  # (num_heads, num_nodes, num_nodes)
        #attn_weights = F.softmax(attn_scores, dim=-1)  # Normalize across the last dimension

        return attn_scores