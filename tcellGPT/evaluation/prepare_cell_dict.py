import argparse
from collections import defaultdict
import numpy as np
from graph_objs.data_utils import prepare_fae, dl_from_gex
import torch
from training.training_utils import get_regression_loss, validate
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
from edge_analysis import print_gene_targets
import pickle

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model performance on gene expression data.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--one_hot_encoder', type=str, required=True, help="Path to the one-hot encoder file.")
    parser.add_argument('--not_in_indices', type=str, required=True, help="Path to the file containing not-in indices.")
    parser.add_argument('--adata', type=str, required=True, help="Path to the AnnData object.")
    parser.add_argument('--response_key', type=str, required=True, help="Key for the response column in adata.obs.")
    parser.add_argument('--selected_gene_names', type=str, required=True, help="Path to the file containing selected gene names.")
    parser.add_argument('--fae', type=str, required=True, help="Path to the fae file.")
    parser.add_argument('--cell_dict_save_path', type=str, required=True, help="Path to save the trained model.")
    parser.add_argument('--gene_dict_save_path', type=str, required=True, help="Path to save the trained model.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load the required data
    model = torch.load(args.model, map_location=device)
    one_hot_encoder = torch.load(args.one_hot_encoder)
    not_in_indices = np.load(args.not_in_indices)
    adata = torch.load(args.adata)
    response_key = args.response_key
    selected_gene_names = np.load(args.selected_gene_names, allow_pickle=True)
    fae = torch.load(args.fae)

    # Prepare responses
    skeys = adata.obs.keys()
    responses = adata.obs[response_key]

    # Shuffle not_in_indices
    np.random.shuffle(not_in_indices)

    fae_not = fae[not_in_indices]
    responses_not = np.array(responses[not_in_indices].values.tolist())
    encoded_responses = one_hot_encoder.transform(responses_not.reshape(-1, 1)).toarray()
    num_cells = fae_not.shape[0]
    num_genes = fae_not.shape[1]

    threshold = 0.5  # Does not matter
    mask_ratio = 0.0
    batch_size = 1
    random = False

    eval_data_loader, eval_edge_index = dl_from_gex(
        fae_not, encoded_responses, np.arange(fae_not.shape[0]),
        num_cells=num_cells, threshold=threshold, device=device, 
        mask_ratio=mask_ratio, batch_size=batch_size, random=random
    )

    cell_dict = {}
    gene_dict = defaultdict(list)
    model.eval()

    for idx, eval_batch in enumerate(tqdm(eval_data_loader, desc="Evaluating")):
        eval_batch = eval_batch.to(device)
        if idx not in cell_dict.keys():
            output, edge_index = validate(model, eval_batch, False)

            mask = torch.ones_like(eval_batch.mask)
            random_gene = np.random.randint(num_genes)
            mask[random_gene] = False

            regression_loss, pred, label = get_regression_loss(
                model, edge_index, eval_batch.y, mask=mask
            )
            gene_dict[selected_gene_names[int(random_gene)]].append(regression_loss.item())

            logits = model.classifier(output)
            prediction = torch.argmax(logits, dim=-1)
            response = torch.argmax(eval_batch.response, axis=-1)
            edge_index = remove_self_loops(edge_index)[0]

            cell_dict[idx] = {
                'response': response.cpu(),
                'embed': output.cpu().numpy(),
                'interactions': print_gene_targets(edge_index, num_genes, selected_gene_names),
                'num_edges': edge_index.shape[1],
                'prediction': prediction.cpu(),
                selected_gene_names[int(random_gene)]: {
                    'loss': regression_loss.item(),
                    'pred': pred.item(),
                    'label': label.item()
                }
            }
    # save cell_dict and gene_dict as pkl files
    with open(args.cell_dict_save_path, 'wb') as f:
        pickle.dump(cell_dict, f)
    with open(args.gene_dict_save_path, 'wb') as f:
        pickle.dump(gene_dict, f)
