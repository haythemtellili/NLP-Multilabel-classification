import logging
import traceback
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from engine import train_fn, eval_fn
from dataset import MultilabelDataset
from model import MultilabelClassifier
from utils import load_data, preprocess_data, get_data_splits, log_metrics


def run():
    # Load data
    data = load_data(config.PATH_DATA)
    # Preprocess data
    data = preprocess_data(data)
    # Split Data into train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(
        data
    )
    label_encoder.save(config.LABEL_ENCODER_PATH)

    train_dataset = MultilabelDataset(X_train.tolist(), y_train.tolist())
    valid_dataset = MultilabelDataset(X_val.tolist(), y_val.tolist())
    test_dataset = MultilabelDataset(X_test.tolist(), y_test.tolist())

    train_data_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=2
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=True, num_workers=1
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=True, num_workers=1
    )

    print("Length of Train Dataloader: ", len(train_data_loader))
    print("Length of Valid Dataloader: ", len(valid_data_loader))
    print("Length of Test Dataloader: ", len(test_data_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_labels = y_train.shape[1]
    model = MultilabelClassifier(n_labels)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model.to(device)
    model = nn.DataParallel(model)

    best_val_loss = 100
    for epoch in tqdm(range(config.EPOCHS)):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        eval_loss, preds, labels = eval_fn(valid_data_loader, model, device)
        auc_score = log_metrics(preds, labels)["auc_micro"]
        print("AUC score: ", auc_score)
        avg_train_loss, avg_val_loss = train_loss / len(
            train_data_loader
        ), eval_loss / len(valid_data_loader)

        print("Average Train loss: ", avg_train_loss)
        print("Average Valid loss: ", avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict(), config.MODEL_PATH)
            print("Model saved as current val_loss is: ", best_val_loss)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(
        level=logging.INFO,
        filename="logs/train_model.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(asctime)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        run()
    except Exception as e:
        print("Exception occured. Check logs.")
        logger.error(f"Failed to run workflow due to error:\n{e}")
        logger.error(traceback.format_exc())