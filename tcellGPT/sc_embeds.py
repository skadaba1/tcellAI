
import copy
import json
import os
from pathlib import Path
import sys
import warnings

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
import gseapy as gp
import display

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

def __main__(model_dir, gene_list):
    set_seed(42)
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    n_hvg = 1200
    n_bins = 51
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    print(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    gene2idx = vocab.get_stoi()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )

    try:
        model.load_state_dict(torch.load(model_file))
        print(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    model.to(device)
        # Retrieve the data-independent gene embeddings from scGPT
    gene_ids = np.array([id for id in gene2idx.values()])
    gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()

    # Filter on the intersection between the Immune Human HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
    gene_embeddings_filtered = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in set(gene_list)}
    print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings_filtered.keys())))

    gene_list = set(gene_list).intersection(set(gene_embeddings_filtered.keys()))
    gene_loci_df = gene_loci_df[gene_loci_df['gene'].isin(gene_list)]
    # remove duplicates
    gene_loci_df = gene_loci_df.drop_duplicates(subset=['gene'])
    display(gene_loci_df.head(10))

    return gene_embeddings_filtered

if __name__ == "__main__":
    model_dir = '/path_to_model'
    gene_list = dict()
    __main__(model_dir, gene_list)