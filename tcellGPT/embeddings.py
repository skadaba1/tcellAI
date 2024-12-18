import torch
import torch.nn as nn
class ShallowAttentionEmbedding(nn.Module):
    '''
    Args:
    - num_embeddings: number of genes to embed
    - embedding_dim: dimension of the embedding vector
    - init_embeddings: pre-trained embeddings for genes

    Returns:
    - embedding: embedding matrix of shape (num_embeddings, embedding_dim)
    '''
    def __init__(self, num_embeddings, embedding_dim, init_embeddings=None):
        super(ShallowAttentionEmbedding, self).__init__()
        # Create embedding via indexing
        if init_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(init_embeddings, freeze=False)
        else:
          self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        indices = (torch.arange(x.size(0), device=self.embedding.weight.device) % self.embedding.num_embeddings).unsqueeze(0)
        out = self.embedding(indices)
        return self.embedding(indices).squeeze(0)


class ShallowGeneEmbedding(nn.Module):
    '''
    Args:
    - input_dim: dimension of the input vector
    - embedding_dim: dimension of the embedding vector
    - init_embeddings: pre-trained embeddings for genes

    Returns:
    - embedding: embedding matrix of shape (input_dim, embedding_dim)
    '''
    def __init__(self, input_dim, embedding_dim, init_embeddings=None):
        super(ShallowGeneEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        self.init_embeddings = init_embeddings

    def forward(self, x):
        if(self.init_embeddings is not None):
          return self.embedding(x) + self.init_embeddings
        else:
          return self.embedding(x)