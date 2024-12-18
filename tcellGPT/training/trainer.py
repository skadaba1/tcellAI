import torch
import torch.distributed as dist
import gc
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb
from training.training_utils import process_batch, validate


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler, 
        train_loader, 
        val_loader, 
        device, 
        num_epochs, 
        effective_batch_size, 
        print_interval,
        reg_nl=1e0, 
        reg_edge=1e-1, 
        reg_ce=1e2,
        num_repeats=1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.effective_batch_size = effective_batch_size
        self.print_interval = print_interval
        self.reg_nl = reg_nl
        self.reg_edge = reg_edge
        self.reg_ce = reg_ce
        self.num_repeats = num_repeats
        
        # Wrap the model in DDP
        self.model = DDP(self.model, device_ids=[device])

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")):
            batch = batch.to(self.device)

            for _ in range(self.num_repeats):
                batch.mask = batch.mask[torch.randperm(batch.mask.shape[0])]
                output, loss, edge_penalties, ce_loss, edge_index = process_batch(self.model, batch)
                total_loss = self.reg_nl * loss + self.reg_edge * edge_penalties + self.reg_ce * ce_loss

                total_loss.backward()

            if (batch_idx + 1) % self.effective_batch_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                wandb.log({
                    'train_loss': total_loss.item(),
                    'node_loss': loss.item(),
                    'edge_loss': edge_penalties.item(),
                    'cross_entropy_loss': ce_loss.item(),
                    'num_edges': edge_index.shape[1]
                })

            del batch, output, loss, edge_penalties, ce_loss, total_loss, edge_index
            torch.cuda.empty_cache()
            gc.collect()

            self.scheduler.step()

            if batch_idx % (self.print_interval * self.effective_batch_size) == 0:
                self.validate()

    def validate(self):
        self.model.eval()

        with torch.no_grad():
            val_batch = next(iter(self.val_loader))
            val_batch = val_batch.to(self.device)

            val_output, val_loss, val_edge_penalties, val_ce_loss, val_edge_index = validate(self.model, val_batch)
            val_total_loss = self.reg_nl * val_loss + self.reg_edge * val_edge_penalties + self.reg_ce * val_ce_loss

            wandb.log({
                'val_loss': val_total_loss.item(),
                'val_node_loss': val_loss.item(),
                'val_edge_loss': val_edge_penalties.item(),
                'val_cross_entropy_loss': val_ce_loss.item(),
                'val_num_edges': val_edge_index.shape[1]
            })

            del val_batch, val_output, val_loss, val_edge_penalties, val_ce_loss, val_total_loss
            torch.cuda.empty_cache()
            gc.collect()

        self.model.train()

    def train(self):
        print(f"Training with weights NL: {self.reg_nl}, CE: {self.reg_ce}, E: {self.reg_edge}")
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)

# Example usage for distributed training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed PyTorch Training")
    parser.add_argument("--model", type=str, required=True, help="Path to the model definition file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--val_dataset", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--optimizer", type=str, required=True, help="Optimizer configuration")
    parser.add_argument("--scheduler", type=str, required=True, help="Scheduler configuration")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--effective_batch_size", type=int, default=32, help="Effective batch size")
    parser.add_argument("--print_interval", type=int, default=10, help="Print interval")
    parser.add_argument("--reg_nl", type=float, default=1e0, help="Node loss regularization weight")
    parser.add_argument("--reg_edge", type=float, default=1e-1, help="Edge loss regularization weight")
    parser.add_argument("--reg_ce", type=float, default=1e2, help="Cross-entropy loss regularization weight")
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repeats for mask permutation")

    args = parser.parse_args()

    dist.init_process_group("nccl")

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Load model, optimizer, scheduler, datasets
    model = torch.load(args.model)
    dataset = torch.load(args.dataset)
    val_dataset = torch.load(args.val_dataset)
    optimizer = torch.optim.Adam(model.parameters()) if args.optimizer == "adam" else torch.optim.SGD(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10) if args.scheduler == "step" else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=DataLoader(dataset, sampler=DistributedSampler(dataset)),
        val_loader=DataLoader(val_dataset, sampler=DistributedSampler(val_dataset)),
        device=local_rank,
        num_epochs=args.num_epochs,
        effective_batch_size=args.effective_batch_size,
        print_interval=args.print_interval,
        reg_nl=args.reg_nl,
        reg_edge=args.reg_edge,
        reg_ce=args.reg_ce,
        num_repeats=args.num_repeats
    )
    trainer.train()

    dist.destroy_process_group()
