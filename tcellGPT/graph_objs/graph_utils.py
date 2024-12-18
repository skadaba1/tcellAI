import torch

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Differentiable sampling using Gumbel-Softmax.
    """
    gumbels = -torch.empty_like(logits).exponential_().log()  # Gumbel noise
    y = (logits + gumbels) / temperature
    y_soft = y.softmax(dim=-1)

    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y

def create_edges_and_weights(attention_matrix, threshold, temperature=0.5, hard=False):
    """
    Use Gumbel-Softmax to create edges and assign weights based on attention values.
    """
    num_nodes = attention_matrix.size(0)
    logits = attention_matrix.view(-1)
    probs = gumbel_softmax(attention_matrix, temperature, hard=hard).view(num_nodes, num_nodes)
    # Create edge_index from probs
    edge_index = (probs > 0.7).nonzero(as_tuple=False).T  # shape (2, num_edges)
    #print(f"Added {edge_index.shape[1]} edges!")

    return edge_index, probs[edge_index[0], edge_index[1]], probs

def custom_add_self_loops(num_nodes, updated_edge_index, edge_weights, probs):
    # Add self-loops to the edge index
  self_loop_index = torch.arange(num_nodes, dtype=torch.long, device=updated_edge_index.device).unsqueeze(0)
  self_loops = torch.cat([self_loop_index, self_loop_index], dim=0)

  # Concatenate the self-loops to updated_edge_index
  updated_edge_index = torch.cat([updated_edge_index, self_loops], dim=1)

  # Add weights for the self-loops (e.g., set them to 1 or some other value)
  self_loop_weights = probs[self_loop_index, self_loop_index].squeeze(0)

  edge_weights = torch.cat([edge_weights, self_loop_weights])
  return updated_edge_index, edge_weights
