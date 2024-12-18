import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import umap
import nx
import gp
from collections import defaultdict

def get_connected_targets(edge_array, source_node):
    """
    Get connected targets for a given source node in an edge array.

    Args:
    - edge_array (numpy.ndarray): Edge array of shape (2, num_edges).
    - source_node (int): Source node for which connected targets are to be found.

    Returns:
    - target_nodes (list): List of connected target nodes.
    """
    # Check if the input array has the correct shape
    assert edge_array.shape[0] == 2, "Edge array should have shape (2, N)"

    # Get indices where the source node matches the given source_node
    target_indices = np.where(edge_array[0, :] == source_node)[0]

    # Retrieve the corresponding target nodes
    target_nodes = edge_array[1, target_indices]

    return target_nodes.tolist()

def print_gene_targets(edges, num_genes, selected_gene_names):
  interactions = set()
  for gene_indx in range(num_genes):
    out = get_connected_targets(edges.cpu().numpy(), gene_indx)
    if(len(out) > 0):
      for i in out:
        interactions.add((selected_gene_names[gene_indx], selected_gene_names[i]))
      #print(f'Gene {selected_gene_names[gene_indx]} connected to: {out_genes}')
  return set(interactions)

def plot_umap_embeddings(cell_dict):
    # Extract embeddings and responses from the cell dictionary
    embeddings = np.array([cell['embed'] for cell in cell_dict.values()])
    embeddings = embeddings.reshape(embeddings.shape[0], -1)
    responses = [cell['response'] for cell in cell_dict.values()]
    responses = ['CR' if response == 0 else 'PD' for response in responses]

    # Determine the minimum count among the response categories
    response_counts = Counter(responses)
    min_count = min(response_counts.values())

    # Create a balanced dataset
    balanced_embeddings = []
    balanced_responses = []
    for response in response_counts:
        indices = [i for i, r in enumerate(responses) if r == response]
        sampled_indices = np.random.choice(indices, size=min_count, replace=False)
        balanced_embeddings.extend(embeddings[sampled_indices])
        balanced_responses.extend([response] * min_count)

    balanced_embeddings = np.array(balanced_embeddings)

    # Use UMAP to reduce the embeddings to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(balanced_embeddings)

    # Create a unique color for each response category
    unique_responses = list(np.unique(balanced_responses))
    colors = plt.cm.tab10(range(len(unique_responses)))
    color_map = {response: colors[i] for i, response in enumerate(unique_responses)}

    # Plot the 2D UMAP projection with colors based on response category
    plt.figure(figsize=(10, 7))
    for response in unique_responses:
        idx = [i for i, r in enumerate(balanced_responses) if r == response]
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    label=response, color=color_map[response], alpha=0.1)

    plt.title("UMAP Projection of Cell Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Response Category")
    plt.show()


def pathway_analysis(interactions):
  '''
  Perform pathway analysis on a set of interactions using KEGG Homo Sapiens.

  Args:
  - interactions: Set of interactions.

  Returns:
  - None
  '''
  try:
    if(len(interactions) > 0):
      unique_genes = set(gene for pair in interactions for gene in pair)

      # Convert set to a list
      gene_list = list(unique_genes)

      # Perform enrichment analysis
      enr = gp.enrichr(
          gene_list=gene_list,
          gene_sets='KEGG_2019_Human',  # Specify the pathway database
          organism='Human',            # Specify organism
      )

      # Convert results to a DataFrame
      results_df = enr.results

      # Select the top enriched pathways for visualization
      top_results = results_df.head(5)  # Adjust the number of pathways as needed

      # Extract genes associated with top pathways
      pathway_gene_mapping = {}
      for _, row in top_results.iterrows():
          pathway_name = row['Term']
          genes = row['Genes'].split(';')  # Genes associated with the pathway
          pathway_gene_mapping[pathway_name] = genes

      # Create the network graph
      G = nx.Graph()

      # Add nodes and edges
      for pathway, genes in pathway_gene_mapping.items():
          G.add_node(pathway, type='pathway')  # Add pathway node
          for gene in genes:
              G.add_node(gene, type='gene')  # Add gene node
              G.add_edge(pathway, gene)  # Add edge between pathway and gene

      # Set node colors based on type
      node_colors = []
      for node in G.nodes(data=True):
          if node[1]['type'] == 'pathway':
              node_colors.append('red')  # Pathway nodes in red
          else:
              node_colors.append('blue')  # Gene nodes in blue

      # Draw the network
      plt.figure(figsize=(5, 4))
      pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
      nx.draw(
          G, pos,
          with_labels=True,
          node_color=node_colors,
          edge_color='gray',
          node_size=1000,
          font_size=10
      )
      plt.title("Gene-Pathway Interaction Network", fontsize=14)
      plt.show()

      gp.barplot(enr.results, title='KEGG Pathway Enrichment')
  except:
    pass

def display_heatmap(data, title="Heatmap", xlabel="X-axis", ylabel="Y-axis", cmap="viridis", colorbar_label="Intensity"):
    """
    Displays a heatmap from a matrix or tensor.

    Parameters:
    - data (numpy.ndarray or torch.Tensor): The 2D matrix or tensor to visualize.
    - title (str): Title of the heatmap.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - cmap (str): Colormap for the heatmap.
    - colorbar_label (str): Label for the colorbar.
    """
    if not isinstance(data, np.ndarray):
        try:
            data = data.cpu().numpy()  # Convert torch.Tensor to numpy if necessary
        except AttributeError:
            raise ValueError("Input data must be a numpy array or a torch tensor.")

    plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect="auto", cmap=cmap)
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def get_edge_lists(cell_dict, classes_all, one_hot_encoder=None):
    '''
    Get edge lists for each class.

    Args:
    - cell_dict (dict): Dictionary containing cell-specific information.
    - classes_all (list): List of all classes.

    Returns:
    - edge_lists (dict): Dictionary of edge lists for each class.
    '''
    classes = one_hot_encoder.categories_[0]
    edge_lists = {c: Counter() for c in classes_all}
    for key in cell_dict.keys():
        edges = cell_dict[key]['interactions']
        label = cell_dict[key]['response']
        # Convert label to string label
        label = one_hot_encoder.categories_[0][label]
        edge_lists[label].update(edges)
    return edge_lists

# Function to check differential upregulation
def find_differentially_upregulated(cell_dict, edge_lists, target_label, other_labels, threshold):
    '''
    Find differentially upregulated edges for a given target label.

    Args:
    - edge_lists (dict): Dictionary of edge lists for each class.
    - target_label (str): Target label for upregulation.
    - other_labels (list): List of other labels to consider.
    - threshold (float): Threshold for differential upregulation.

    Returns:
    - upregulated (set): Set of differentially upregulated edges.
    '''
    num_each = get_num_each(cell_dict, ['CR', 'PR', 'SD', 'PD'])
    other_labels = set([label for label in other_labels if num_each[label] > 0])
    if(num_each[target_label] <= 0):
        return set()
    upregulated = set()
    all_edges = set()
    for label, counter in edge_lists.items():
        if label != target_label:
            all_edges.update(counter.keys())

    for edge in all_edges:
        num_in_target = edge_lists[target_label][edge]/num_each[target_label]
        num_in_others = sum([edge_lists[label][edge]/num_each[label] for label in other_labels]) / len(other_labels)
        if num_in_target/num_in_others > threshold:
            upregulated.add(edge)

    return upregulated

def find_unique_edges_soft(cell_dict, edge_lists, target_label, other_labels, threshold):
  '''
  Find unique/significant edges for a given target label.

  Args:
  - cell_dict: Dictionary containing cell-specific information.
  - edge_lists: Dictionary of edge lists for each class.
  - target_label: Target label for unique edges.
  - other_labels: List of other labels to consider.
  - threshold: Threshold for unique edges.

  Returns:
  - unique_edges: Set of unique/significant edges.
  '''

  num_each = get_num_each(cell_dict, ['CR', 'PR', 'SD', 'PD'])
  if(num_each[target_label] <= 0):
    return set()
  # to be unique, edge must exist in > threshold of target_label and < 1 - threshold of other labels
  unique_edges = set()
  all_edges = set()
  for label, counter in edge_lists.items():
    all_edges.update(counter.keys())

  for edge in all_edges:
    num_in_target = edge_lists[target_label][edge] >= threshold * num_each[target_label]
    num_in_others = [edge_lists[label][edge] <= num_each[label]*(1-threshold) for label in other_labels]
    if(num_in_target and all(num_in_others)):
      unique_edges.add(edge)

  return unique_edges

def find_unique_edges(target_list, other_lists):
    return target_list.difference(*other_lists)

def get_num_each(cell_dict, classes_all, one_hot_encoder=None):
  classes = one_hot_encoder.categories_[0]
  num_each = {c:0 for c in classes_all}
  for key in cell_dict.keys():
    class_name = one_hot_encoder.categories_[0][cell_dict[key]['response']]
    num_each[class_name] += 1
  return num_each

def get_k_largest_indices(matrix, k):
    # Get the largest elements and their indices
    largest_values, indices = torch.topk(matrix.flatten(), k)

    # Reshape the indices to the 2D matrix
    indices_2d = torch.stack((indices // matrix.size(1), indices % matrix.size(1)), dim=1)

    return indices_2d

# average attention matrices over each cell type
def analyze_attentions(cell_dict, num_genes, k, classes_all, selected_gene_names=None, one_hot_encoder=None):
  '''
  Analyze significant edges that emerge from pooled attention matrices in each cohort

  Args:
  - cell_dict: Dictionary containing cell-specific information.
  - num_genes: Number of genes.
  - k: Number of top edges to consider.
  - classes_all: List of all classes.

  Returns:
  - top_k_dict: Dictionary of top k edges for each class.
  '''
  classes = one_hot_encoder.categories_[0]
  attention_matrices = {i: torch.zeros((num_genes, num_genes)) for i in classes}

  for key in cell_dict.keys():
    class_name = one_hot_encoder.categories_[0][cell_dict[key]['response']]
    attention_matrices[class_name] += cell_dict[key]['attention'].cpu()

  # Get top 10 in each attention matrix
  k = 10
  top_k_dict = defaultdict()
  for target_name, attention_matrix in attention_matrices.items():
    attention_matrix_others = torch.sum(torch.stack([matrix for class_name, matrix in attention_matrices.items() if class_name!=target_name]), dim=0)
    k_largest = set([tuple(item) for item in get_k_largest_indices(attention_matrix - attention_matrix_others, k=k).tolist()])
    top_k_dict[target_name] = set([(selected_gene_names[i], selected_gene_names[i])  for i, j in k_largest])
  for c in classes_all:
    if(c not in top_k_dict):
      top_k_dict[c] = set()
  return top_k_dict