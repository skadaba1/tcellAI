
from sklearn.metrics import classification_report, accuracy_score
import torch
from losses import loss_fn, loss_fn_ce
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, pearsonr
import numpy as np

def eval_native_classifier(cell_dict):
    '''
    Evaluate the native classifier on the cell dictionary

    Args:
    - cell_dict (dict): Dictionary containing cell-specific information.

    Returns:
    - report (str): Classification report.
    '''

    # model.eval()
    y_true = []
    y_pred = []

    for cell_idx, cell_data in cell_dict.items():
        # Extract the 1-hot encoded response and embedding
        true_class = cell_data['response']
        predicted_class = cell_data['prediction'] #np.argmax(logits, axis=-1)


        # Append true and predicted classes to lists
        y_true.append(true_class)
        y_pred.append(predicted_class)

    # Generate the classification report
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    return report

def get_regression_loss(model, edge_index, labels, mask=None):
  '''
  Get the regression loss for a given edge index and labels by computing edge_index over nodes.

  Args:
  - model: PyTorch model.
  - edge_index (torch.Tensor): Edge index of shape (2, num_edges).
  - labels (torch.Tensor): Labels of shape (num_edges,).

  Returns:
  - loss (torch.Tensor): Regression loss.
  '''
  x = model.x_final
  attention_init = model.attention_final
  model.eval()
  loss = 0.0
  pred = None
  with torch.no_grad():
    if(mask is not None):
      num_masked_nodes = mask.numel() - mask.sum().item()
      for node_idx in (~mask).nonzero(as_tuple=True)[0]:
          # Pool the neighborhood of the masked node

          neighbor_idxs = edge_index[1][edge_index[0] == node_idx]  # Get neighbors of the masked node
          # print(neighbor_idxs.shape)

          if neighbor_idxs.numel() == 0:  # No neighbors

              # Assign a high differentiable penalty loss for isolated nodes
              node_loss = loss_fn(x*0, labels[node_idx].unsqueeze(0))
          else:

              # Compute the neighborhood embedding if neighbors exist
              neighbor_embeddings = x[neighbor_idxs]

              # apply attention from attention_init and sum
              weights = F.softmax(attention_init[node_idx, neighbor_idxs].unsqueeze(0), dim=1)
              pooled_embedding = (weights @ neighbor_embeddings)
              if torch.isnan(pooled_embedding).any():
                  # Handling NaN values
                  continue

              # Make prediction for the current masked node
              pred = model.regressor(pooled_embedding)  # Adding batch dimension for single node

              # Compute the loss for the current masked node
              node_loss = loss_fn(pred, labels[node_idx].unsqueeze(0))

          # Accumulate the loss for each masked node
          loss += node_loss if not torch.isnan(node_loss) else 0.0
      loss /= num_masked_nodes if num_masked_nodes > 0 else 1
      return loss, pred, labels[node_idx]

# Regression evaluation and analysis
def plot_and_score_r2(cell_dict, selected_gene_names=None):
    '''
    Plot and score R^2 and Pearson correlation coefficient.

    Args:
    - cell_dict: Dictionary containing cell-specific information.

    Returns:
    - None
    '''
    y_hat = []
    y = []

    # Extract predictions and labels
    for idx in cell_dict.keys():
        for key in cell_dict[idx].keys():
            if key in selected_gene_names:
                y_hat.append(cell_dict[idx][key]['pred'])
                y.append(cell_dict[idx][key]['label'])

    # Compute R^2 score
    r2 = r2_score(y, y_hat)

    # Compute Pearson correlation coefficient
    pearson_corr, _ = pearsonr(y, y_hat)

    # Print metrics
    print(f"R^2 Score: {r2}")
    print(f"Pearson Correlation Coefficient: {pearson_corr}")

    # Scatter plot of predictions and observations
    plt.figure(figsize=(8, 8))  # Make the plot larger for clarity
    plt.scatter(y, y_hat, s=10, alpha=0.7, label="Data points")  # Smaller dots with some transparency

    # Compute line of best fit
    coefficients = np.polyfit(y, y_hat, 1)  # Fit a line (degree 1 polynomial)
    best_fit_line = np.poly1d(coefficients)
    y_fit = best_fit_line(y)

    # Plot the line of best fit
    plt.plot(y, y_fit, color="red", linewidth=2, label=f"Best fit line (y = {coefficients[0]:.2f}x + {coefficients[1]:.2f})")

    # Adjust axes limits and set integer increments
    min_val = int(min(min(y), min(y_hat)))
    max_val = int(max(max(y), max(y_hat)))
    buffer = 1  # Add a buffer of 1 for spacing

    plt.xlim(min_val - buffer, max_val + buffer)
    plt.ylim(min_val - buffer, max_val + buffer)
    plt.xticks(range(min_val - buffer, max_val + buffer + 1, 1))  # Integer ticks
    plt.yticks(range(min_val - buffer, max_val + buffer + 1, 1))  # Integer ticks

    # Add labels, legend, and title
    plt.xlabel('Gene expression (NRC)')
    plt.ylabel('Predicted gene expression (NRC)')
    plt.title(f'Predicted vs. observed normalized gene expression \n\nR^2: {r2:.2f}, Pearson: {pearson_corr:.2f}')
    plt.legend()
    plt.grid(alpha=0.3)  # Add a light grid for better readability
    plt.tight_layout()  # Ensure proper spacing in the plot
    plt.show()

def plot_extreme_values(data_dict, k, mode):
    for key in data_dict:
      data_dict[key] = np.mean(data_dict[key])
    """
    Plots a bar graph of the k highest and k smallest values in the dictionary.

    Parameters:
        data_dict (dict): Dictionary with keys as labels and values as numerical data.
        k (int): Number of highest and smallest values to display.
    """
    # Sort the dictionary by values
    sorted_items = sorted(data_dict.items(), key=lambda x: x[1])

    # Extract k smallest and k largest
    smallest_k = sorted_items[:k]
    largest_k = sorted_items[-k:]

    # Combine for plotting
    if(mode == 'smallest'):
      extreme_values = smallest_k
    elif(mode == 'largest'):
      extreme_values = largest_k
    else:
      raise Exception("Invalid mode")

    # Unpack labels and values
    labels, values = zip(*extreme_values)

    # Create bar plot
    x_pos = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar(x_pos, values, color="skyblue")
    plt.xticks(x_pos, labels, rotation=45, ha="right")
    plt.title(f"Top {k} {mode} per-gene MSE values")
    plt.xlabel("Gene")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.show()