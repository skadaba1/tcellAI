import torch.nn.functional as F
import torch.nn as nn

# Example loss function for masked nodes
def loss_fn(pred, target):
    return F.mse_loss(pred, target)

# cross entropy loss function form
def loss_fn_ce(pred, target, label_smoothing=0.0):
    return F.cross_entropy(pred, target, label_smoothing=label_smoothing) #weight=torch.tensor([3, 2], device=pred.device)

loss_fn_ce = nn.CrossEntropyLoss()