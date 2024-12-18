import wandb
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import gc
import torch

def validate(model, batch, compute_loss=True):
  """
  Run validation on a given batch.

  Args:
  - model: PyTorch model.
  - batch: Batch of data.
  - compute_loss: Whether to compute the loss or just return computed neighborhoods

  Returns:
  - output: Model output.
  - loss: Loss value.
  - additional based on arguments
  """
  model.eval()
  with torch.no_grad():
    if(not compute_loss):
      output, edge_index = process_batch(model, batch, compute_loss)
    else:
      output, loss, edge_penalities, ce_loss, edge_index = process_batch(model, batch, compute_loss)

  model.train()
  if(not compute_loss):
    return output, edge_index
  return output, loss, edge_penalities, ce_loss, edge_index

def process_batch(model, batch, compute_loss=True):
    """
    Process batch for training step.

    Args:
    - model: PyTorch model.
    - batch: Batch of data.
    - compute_loss: Whether to compute the loss or just return computed neighborhoods

    Returns:
    - model output
    - loss (all losses)
    """
    # Move data to the device
    # Each `batch` contains batched data with attributes: batch.x, batch.edge_index, batch.batch
    x = batch.x  # Node features
    edge_index = batch.edge_index  # Batched edge indices
    batch_indices = batch.batch  # Batch index for each node
    mask = batch.mask
    labels = batch.y
    response = batch.response
    # Pass through the model
    if(not compute_loss):
      output, edge_index = model(x, edge_index, batch_indices, mask=mask, labels=labels, response=response, compute_loss=compute_loss, return_edges=True)
      return output, edge_index
    else:
      output, loss, edge_penalities, ce_loss, edge_index = model(x, edge_index, batch_indices, mask=mask, labels=labels, response=response, compute_loss=compute_loss, return_edges=True)
      return output, loss, edge_penalities, ce_loss, edge_index

def get_splits(num_cells, train_ratio, val_ratio, test_ratio):

  # Calculate the sizes for each split
  train_size = int(train_ratio * num_cells)
  val_size = int(val_ratio * num_cells)
  test_size = num_cells - train_size - val_size  # Adjust for any rounding issues

  # Generate random train, validation, and test indices
  train_indices, val_indices, test_indices = torch.utils.data.random_split(
      range(num_cells), [train_size, val_size, test_size]
  )
  return train_indices, val_indices, test_indices

import networkx as nx
import matplotlib.pyplot as plt

def visualize_edges(edge_index):
  """
  Visualizes the edges in a graph given the edge_index matrix.

  Args:
  - edge_index (torch.Tensor): Edge index of shape (2, num_edges).

  Returns:
  - N/A
  """
  # Create a directed graph (can change to undirected if needed)
  G = nx.Graph()  # Use nx.Graph() if the graph is undirected

  # Add edges from edge_index
  for src, dst in edge_index.T:
      G.add_edge(src, dst)

  # Draw the graph
  plt.figure(figsize=(10, 8))
  pos = nx.spring_layout(G)  # Layout for a nicer visualization
  nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, font_color="white")
  nx.draw_networkx_edge_labels(G, pos, edge_labels={(src, dst): f"{src}->{dst}" for src, dst in G.edges()})
  plt.show()

# Custom cosine annealing scheduler with a minimum learning rate
class CosineAnnealingWithMinLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_hold_steps, num_training_steps, init_lr=1e-3, min_lr=5e-4, last_epoch=-1):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return (float(current_step) / float(max(1, num_warmup_steps)))
            if current_step < num_warmup_steps + num_hold_steps:
                return 1.0
            progress = float(current_step - (num_warmup_steps + num_hold_steps)) / float(max(1, num_training_steps - (num_warmup_steps + num_hold_steps)))
            cosine_decay_lr = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * progress)))
            return max(min_lr/init_lr, cosine_decay_lr)

        super(CosineAnnealingWithMinLRScheduler, self).__init__(optimizer, lr_lambda, last_epoch)
