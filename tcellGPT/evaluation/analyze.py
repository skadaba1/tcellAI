import argparse
import pickle
from edge_analysis import *
from objectives_analysis import *

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Analyze attentions and perform pathway analysis.")
    parser.add_argument("--cell_dict", type=str, required=True, help="Path to the cell dictionary .pkl file.")
    parser.add_argument("--gene_dict", type=str, required=True, help="Path to the gene dictionary .pkl file.")
    parser.add_argument("--num_genes", type=int, required=True, help="Number of genes to analyze.")

    args = parser.parse_args()

    # Load dictionaries
    cell_dict = load_pickle(args.cell_dict)
    gene_dict = load_pickle(args.gene_dict)
    num_genes = args.num_genes

    # Perform analysis
    top_k_dict = analyze_attentions(cell_dict, num_genes, k=1000, classes_all=['CR', 'PR', 'SD', 'PD'])
    edge_list_r1 = top_k_dict['CR']
    edge_list_r2 = top_k_dict['PR']
    edge_list_r3 = top_k_dict['SD']
    edge_list_r4 = top_k_dict['PD']

    attention_unique_r1 = find_unique_edges(edge_list_r1, [edge_list_r2, edge_list_r3, edge_list_r4])
    attention_unique_r2 = find_unique_edges(edge_list_r2, [edge_list_r1, edge_list_r3, edge_list_r4])
    attention_unique_r3 = find_unique_edges(edge_list_r3, [edge_list_r1, edge_list_r2, edge_list_r4])
    attention_unique_r4 = find_unique_edges(edge_list_r4, [edge_list_r1, edge_list_r2, edge_list_r3])

    print(attention_unique_r1)
    print(attention_unique_r2)
    print(attention_unique_r3)
    print(attention_unique_r4)

    threshold = 5
    edge_lists = get_edge_lists(cell_dict, ['CR', 'PR', 'SD', 'PD'])
    upregulated_r1 = find_differentially_upregulated(edge_lists, 'CR', ['PR', 'SD', 'PD'], threshold)
    upregulated_r2 = find_differentially_upregulated(edge_lists, 'PR', ['CR', 'SD', 'PD'], threshold)
    upregulated_r3 = find_differentially_upregulated(edge_lists, 'SD', ['CR', 'PR', 'PD'], threshold)
    upregulated_r4 = find_differentially_upregulated(edge_lists, 'PD', ['CR', 'PR', 'SD'], threshold)
    print(upregulated_r1)
    print(upregulated_r2)
    print(upregulated_r3)
    print(upregulated_r4)

    # Find unique genes in responding vs non-responding
    edge_lists = get_edge_lists(cell_dict, ['CR', 'PR', 'SD', 'PD'])
    threshold = 0.2
    unique_in_r1 = find_unique_edges_soft(cell_dict, edge_lists, 'CR', ['PR', 'SD', 'PD'], threshold)
    unique_in_r2 = find_unique_edges_soft(cell_dict, edge_lists, 'PR', ['CR', 'SD', 'PD'], threshold)
    unique_in_r3 = find_unique_edges_soft(cell_dict, edge_lists, 'SD', ['PR', 'CR', 'PD'], threshold)
    unique_in_r4 = find_unique_edges_soft(cell_dict, edge_lists, 'PD', ['PR', 'SD', 'CR'], threshold)

    print(unique_in_r1)
    print(unique_in_r2)
    print(unique_in_r3)
    print(unique_in_r4)

    pathway_analysis(attention_unique_r1)
    pathway_analysis(attention_unique_r4)

    pathway_analysis(upregulated_r1)
    pathway_analysis(upregulated_r4)

    pathway_analysis(unique_in_r1)
    pathway_analysis(unique_in_r4)

    eval_native_classifier(cell_dict)
    plot_and_score_r2(cell_dict)
    plot_extreme_values(gene_dict, 10, 'smallest')
    plot_extreme_values(gene_dict, 10, 'largest')
