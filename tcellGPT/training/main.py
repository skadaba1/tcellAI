import argparse
import itertools
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch
from training.trainer import Trainer
from graph_objs.data_utils import prepare_fae, dl_from_gex, get_splits
from graph_objs.model import DynamicGAT
from training.training_utils import CosineAnnealingWithMinLRScheduler
import wandb

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(description="Train a DynamicGAT model.")
    parser.add_argument("--sc_embed", type=str, required=True, help="Path to single-cell embedding file.")
    parser.add_argument("--adata", type=str, required=True, help="Path to the anndata file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--fae", type=str, required=True, help="Path to the fae file.")
    parser.add_argument("--responses_fae", type=str, required=True, help="Path to the responses_fae file.")
    args = parser.parse_args()

    # Load the arguments from dict
    sc_embed_path = args.sc_embed
    adata_path = args.adata
    model_save_path = args.model_path

    # 2389 genes
    torch.set_printoptions(threshold=10000)  # Set threshold to a high number to display full tensor

    # Load sc_embed and adata (assuming they are pre-saved files)
    sc_embed = torch.load(sc_embed_path)
    adata = torch.load(adata_path)
    fae = torch.load(args.fae)
    responses_fae = torch.load(args.responses_fae)

    # randomly sample rows from fae
    num_samples = 60000
    fae_sampled, responses_one_hot_sampled, indices, not_in_indices, patients_sampled, not_in_patients, one_hot_encoder = prepare_fae(num_samples, fae, responses_fae)

    # Response_30d, Response_3m, Response_6m
    keys = adata.obs.keys()

    num_cells = fae_sampled.shape[0]
    num_genes = fae_sampled.shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sc_embed = sc_embed.to(device)

    # Initialize GAT model
    batch_size = 1
    effective_batch_size = 16
    n_features = 4
    embedding_dim = 512
    hidden_dim = 512
    output_dim = 512
    num_layers = 4
    lr = 5e-4
    num_epochs = 1
    heads = 4
    print_interval = 10
    threshold = 0.5  # correlation threshold for init edges
    att_threshold = 0.3  # 1/(num_genes**2) arbitrary
    reg_edge = 0.0
    reg_ce = 1e0
    reg_nl = 1e0
    n_classes = 2
    num_sparse_connections = min(512, num_genes)
    dropout_prob = 0.1
    random = False
    negative_slope = 0.02
    mask_ratio = 0.3
    n_repeats = 1

    # Get train, test split indices for # cells
    train_idx, val_idx, test_idx = get_splits(num_cells, 0.8, 0.05, 0.15)

    train_data_loader, init_train_edge_index = dl_from_gex(fae_sampled, responses_one_hot_sampled, train_idx, num_cells=num_cells, threshold=threshold, device=device, mask_ratio=mask_ratio, batch_size=batch_size, random=random)
    val_data_loader, init_val_edge_index = dl_from_gex(fae_sampled, responses_one_hot_sampled, val_idx, num_cells=num_cells, threshold=threshold, device=device, mask_ratio=mask_ratio, batch_size=batch_size, random=random)
    val_data_loader = itertools.cycle(val_data_loader)
    test_data_loader, init_test_edge_index = dl_from_gex(fae_sampled, responses_one_hot_sampled, test_idx, num_cells=num_cells, threshold=threshold, device=device, mask_ratio=mask_ratio, batch_size=batch_size, random=random)
    print(init_train_edge_index.shape, init_val_edge_index.shape, init_test_edge_index.shape)

    model = DynamicGAT(num_genes=num_genes, in_channels=n_features, embedding_dim=embedding_dim, hidden_channels=hidden_dim, out_channels=output_dim, num_layers=num_layers, n_classes=n_classes, threshold=att_threshold, heads=heads, num_sparse_connections=num_sparse_connections, negative_slope=negative_slope, dropout_prob=dropout_prob).to(device)
    # Print total number of trainable parameters
    print(f"Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Training on patients {patients_sampled}")
    print(f"Evaluating on patients: {not_in_patients}")

    num_training_steps = num_epochs * len(train_data_loader)
    num_warmup_steps = int(0.05 * num_training_steps)  # 1% warmup
    num_hold_steps = int(0.1 * num_training_steps)  # 10% hold
    min_lr = 1e-5  # Minimum learning rate you want to set

    scheduler = CosineAnnealingWithMinLRScheduler(optimizer, num_warmup_steps, num_hold_steps, num_training_steps, min_lr=min_lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        device=device,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        print_interval=print_interval,
        reg_nl=reg_nl,
        reg_edge=reg_edge,
        reg_ce=reg_ce,
        num_repeats=n_repeats
    )
    wandb.login()
    wandb.init()
    trainer.train()
    trainer.validate()

    # Save model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    wandb.finish()
