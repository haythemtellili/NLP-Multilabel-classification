import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm


def loss_fn(outputs, labels, class_weights):
    class_weights_tensor = torch.Tensor(np.array(list(class_weights.values())))
    if labels is None:
        return None
    return nn.BCEWithLogitsLoss(weight=class_weights_tensor)(outputs, labels.float())


def train_fn(data_loader, model, optimizer, device, scheduler, class_weights):
    """
    Function to train the model
    """
    train_loss = 0.0
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]
        mask = d["mask"]
        targets = d["labels"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)

        loss = loss_fn(outputs, targets, class_weights)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        scheduler.step()
    return train_loss


def eval_fn(data_loader, model, device, class_weights):
    """
    Function to evaluate the model
    """
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets, class_weights)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))

    return eval_loss, fin_outputs, fin_targets
