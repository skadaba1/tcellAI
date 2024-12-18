import anndata as ad
import scanpy as sc
import pandas as pd
import pyensembl
import numpy as np
import torch

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_pca(features, labels):
    # Flatten the feature array to 2D (samples, features)
    flattened_features = features.reshape(features.shape[0], -1)

    # Perform PCA for dimensionality reduction to 2D
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(flattened_features)

    # Plot the PCA results
    plt.figure(figsize=(10, 7))
    for label in np.unique(labels):
        plt.scatter(
            pca_result[labels == label, 0],
            pca_result[labels == label, 1],
            label=f"Label {label}",
            alpha=0.5,
            s=10
        )
    plt.title("PCA Projection of Features Colored by Labels")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.legend()
    plt.show()


class GeneDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.adata = None
        self.adata_sc = None
        self.gene_loci_df = None
        self.top_n_genes = None
        self.fae = None

    def load_anndata(self):
        self.adata = ad.read_h5ad(self.file_path)
        return set(self.adata.var_names)

    def preprocess_data(self, response_key, n=1200):
        self.adata_sc = sc.read_h5ad(self.file_path)
        sc.pp.normalize_total(self.adata_sc, target_sum=1e4)
        sc.pp.log1p(self.adata_sc)
        sc.tl.rank_genes_groups(self.adata_sc, groupby=response_key, method='t-test')

        de_results = self.adata_sc.uns['rank_genes_groups']
        k = n // 2
        self.top_n_genes = {
            group: [
                self.adata_sc.var_names[int(idx)] for idx in de_results['names'][group][:k]
            ] for group in de_results['names'].dtype.names
        }
        flattened_genes = [gene for sublist in self.top_n_genes.values() for gene in sublist]
        self.top_n_genes = {'gene': flattened_genes}
        return self.top_n_genes

    def extract_gene_loci(self, ensembl_release=77):
        data = pyensembl.EnsemblRelease(ensembl_release)
        gene_names = self.top_n_genes['gene']
        gene_loci_list = []

        for gene in gene_names:
            try:
                ensembl_gene = data.genes_by_name(gene)[0]
                gene_loci_list.append({
                    'gene': gene,
                    'chromosome': ensembl_gene.contig,
                    'start': ensembl_gene.start,
                    'end': ensembl_gene.end,
                })
            except Exception:
                pass

        self.gene_loci_df = pd.DataFrame(gene_loci_list)
        self.gene_loci_df['chromosome'] = self.gene_loci_df['chromosome'].replace({
            'X': 23, 'Y': 23, 'MT': 24
        }).apply(pd.to_numeric, errors='coerce').fillna(0)
        self.gene_loci_df.sort_values(by='start', ascending=True, inplace=True)
        return self.gene_loci_df

    def extract_selected_genes(self):
        gene_list = self.gene_loci_df['gene']
        adata_selected_genes = self.adata[:, self.adata.var_names.isin(gene_list)]
        return adata_selected_genes

    def prepare_data_matrices(self, adata_selected_genes, gene_embeddings_filtered):
        sc_embed = [gene_embeddings_filtered[i] for i in adata_selected_genes.var_names]
        sc_embed = torch.tensor(np.array(sc_embed))
        self.fae = adata_selected_genes.X.toarray()

        return self.fae, sc_embed

    def align_gene_loci_with_selected_genes(self, selected_gene_names):
        gene_loci_df_mod = self.gene_loci_df.set_index("gene")
        gene_loci_df_mod = gene_loci_df_mod.loc[selected_gene_names].reset_index(drop=True)

        gene_loci_df_mod['chromosome'] = pd.to_numeric(gene_loci_df_mod['chromosome'], errors='coerce').fillna(-1)
        gene_loci_data_mod = gene_loci_df_mod.values.astype(float)
        return gene_loci_data_mod

    def augment_and_normalize_data(self, gene_loci_data_mod):
        fae_expanded = np.expand_dims(self.fae, axis=2)
        gene_loci_expanded = np.tile(gene_loci_data_mod, (self.fae.shape[0], 1, 1))
        self.fae = np.concatenate((fae_expanded, gene_loci_expanded), axis=2) + 1e-6

        self.fae[:, :, 2] = (self.fae[:, :, 2] - np.mean(self.fae[:, :, 2])) / np.std(self.fae[:, :, 2])
        self.fae[:, :, 3] = (self.fae[:, :, 3] - np.mean(self.fae[:, :, 3])) / np.std(self.fae[:, :, 3])

        return self.fae

if __name__ == "__main__":
    # Usage Example
    file_path = '/content/drive/MyDrive/raw_T_cell_IP_combined_DGH.h5ad'
    response_key = 'Response_6m'
    gene_embeddings_filtered = ...  # Define your embeddings

    processor = GeneDataProcessor(file_path)
    genes_ccr = processor.load_anndata()
    top_n_genes = processor.preprocess_data(response_key)
    gene_loci_df = processor.extract_gene_loci()
    adata_selected_genes = processor.extract_selected_genes()
    fae, sc_embed = processor.prepare_data_matrices(adata_selected_genes, gene_embeddings_filtered)
    selected_gene_names = adata_selected_genes.var_names
    gene_loci_data_mod = processor.align_gene_loci_with_selected_genes(selected_gene_names)
    fae = processor.augment_and_normalize_data(gene_loci_data_mod)


    # Generate synthetic data for demonstration purposes
    features = fae
    responses_fae = adata_selected_genes.obs[response_key]
    labels = responses_fae
    plot_pca(features, labels)

    print(f"Processed FAE shape: {fae.shape}")
