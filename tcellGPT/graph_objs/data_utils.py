
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops
import torch.nn as nn
from tqdm import tqdm

# Function to initialize a random edge_index for a single cell graph of genes
def initialize_edge_index(expression_matrix, threshold=0.5, random=False):
    """
    Create an edge index based on gene correlations across all cells.

    Args:
    - expression_matrix (torch.Tensor): Expression matrix of shape (num_cells, num_genes).
    - threshold (float): Correlation threshold for creating edges.

    Returns:
    - edge_index (torch.Tensor): Edge index of shape (2, num_edges).
    """

    # if random, create random edge list
    if(random):
        num_cells, num_genes = expression_matrix.shape
        edge_index = torch.randint(low=0, high=num_genes, size=(2, num_cells))
        return edge_index

    num_cells, num_genes = expression_matrix.shape
    expression_mean = expression_matrix.mean(dim=0, keepdim=True)
    expression_std = expression_matrix.std(dim=0, keepdim=True)

    # Normalize expression matrix
    normalized_matrix = (expression_matrix - expression_mean) / (expression_std + 1e-8)

    # Compute correlation between genes
    correlation_matrix = torch.mm(normalized_matrix.t(), normalized_matrix) / num_cells

    # Apply threshold to create adjacency matrix
    adjacency_matrix = correlation_matrix > threshold
    adjacency_matrix.fill_diagonal_(False)  # Remove self-loops

    # Find indices where correlation exceeds threshold
    edge_sources, edge_targets = adjacency_matrix.nonzero(as_tuple=True)
    edge_index = torch.stack([edge_sources, edge_targets], dim=0)


    return edge_index

# Function to create node mask and labels for prediction
def create_mask_and_labels(cell_data_embedded, num_genes, mask_ratio=0.30):
    """
    Generate a mask and labels for nodes where predictions are needed.

    Args:
    - num_genes (int): Number of nodes (genes).
    - mask_ratio (float): Ratio of nodes to be masked for prediction.

    Returns:
    - mask (torch.Tensor): Boolean tensor indicating masked nodes.
    - labels (torch.Tensor): Labels for the masked nodes.
    """
    mask = torch.ones(num_genes, dtype=torch.bool)
    num_masked_nodes = int(num_genes * mask_ratio)
    masked_indices = torch.randperm(num_genes)[:num_masked_nodes]
    mask[masked_indices] = False

    # Generate random labels for masked nodes (for demonstration, replace with real labels)
    labels = cell_data_embedded[:,0]  # Assuming binary labels; modify as needed
    labels[~mask] = -1  # Set labels for unmasked nodes to -1 (ignore index)

    return mask, labels

# Convert each cell to a PyG Data object
def create_graph_data(cell_data, response, edge_index, mask_ratio=0.30):
    """

    Args:
    - cell_data (torch.Tensor): Gene expression data for a single cell.
    - edge_index (torch.Tensor): Edge index of shape (2, num_edges).
    - response (float): Response value for the cell (SD, CR, PR, PD).

    Returns:
    - graph_data (Data): PyG Data object representing the cell graph.
    """
    # Embed the cell data (gene features)
    mask_ratio = mask_ratio
    cell_data_embedded = cell_data.unsqueeze(-1)  # Shape: [2000, 1]

    # Generate mask and labels for this cell's graph
    mask, labels = create_mask_and_labels(cell_data_embedded, cell_data.shape[0], mask_ratio)

    # Create Data object with additional attributes
    response = torch.tensor(response, dtype=torch.float32).unsqueeze(0)

    graph_data = Data(x=cell_data_embedded, edge_index=edge_index, mask=mask, y=labels, response=response)
    return graph_data


def dl_from_gex(fae, responses, indices, num_cells=50000, threshold=0.5, device=None, mask_ratio=0.3, batch_size=1, random=False):
  """
  Creates data loaders from gene expression graph data.

  Args:
  - fae: sparse gene expression matrix
  - responses: classification label of each cell
  - indices: based on split

  Returns:
  - data_loader: data loader for batching
  """
  # Assuming data shape (85000, 2000), where each row represents a cell and each column is a gene feature
  data = torch.Tensor(fae[indices, :, :])  # Replace with actual data

  responses = responses[indices, :]
  graph_dataset = []
  num_genes = data.shape[1]
  edge_index = initialize_edge_index(data[:,:,0], threshold, random)

  #sample_indices = np.random.permutation(data.shape[0])
  for i in tqdm(range(data.shape[0])):  # Iterate over each cell
      cell_data = data[i]  # Shape: [2000] for one cell's gene expression
      graph_data = create_graph_data(cell_data, responses[i], edge_index, mask_ratio=mask_ratio)
      graph_dataset.append(graph_data)
    # Create DataLoader for batching

  data_loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
  return data_loader, edge_index