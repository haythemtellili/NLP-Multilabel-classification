import os
import json
import logging
import itertools
import traceback
import warnings
import numpy as np
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
from utils import load_data, preprocess_data, get_data_splits, log_metrics, set_seeds

def run():
    """ Main function to excute the pipeline of training """
    # Set seeds
    set_seeds()
    # Load data
    data = load_data(config.PATH_DATA)
    # Preprocess data
    data = preprocess_data(data)
    # Split Data into train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = get_data_splits(
        data
    )
    # Create models directory
    if not os.path.exists(config.PATH_MODELS):
        os.makedirs(config.PATH_MODELS)
    # Save Label encoder
    label_encoder.save(os.path.join(config.PATH_MODELS, config.LABEL_ENCODER_PATH))
    # Create datasets
    train_dataset = MultilabelDataset(X_train.tolist(), y_train.tolist())
    valid_dataset = MultilabelDataset(X_val.tolist(), y_val.tolist())
    test_dataset = MultilabelDataset(X_test.tolist(), y_test.tolist())
    # Create DataLaoders 
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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Init the model
    n_labels = y_train.shape[1]
    model = MultilabelClassifier(n_labels)
    # Set the optimizer and the scheduler
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
    # Define Class weights
    train_tags = list(itertools.chain.from_iterable(data.target.values))
    counts = np.bincount([label_encoder.class_to_index[class_] for class_ in train_tags])
    class_weights = {i: 1.0/count for i, count in enumerate(counts)}
    # start training 
    best_val_loss = 100
    for _ in tqdm(range(config.EPOCHS)):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler, class_weights)
        eval_loss, preds, labels = eval_fn(valid_data_loader, model, device, class_weights)
        performance = log_metrics(preds, labels)
        print("Performance: ", performance)

        avg_train_loss, avg_val_loss = train_loss / len(
            train_data_loader
        ), eval_loss / len(valid_data_loader)

        print("Average Train loss: ", avg_train_loss)
        print("Average Valid loss: ", avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.module.state_dict(), os.path.join(config.PATH_MODELS, config.BERT_PATH))
            print("Model saved as current val_loss is: ", best_val_loss)

    print("Start Testing and Finding Threshold")
    _, preds, labels = eval_fn(test_data_loader, model, config.DEVICE)
    testing_result = log_metrics(preds, labels)
    print("Performance on the testing data: ", testing_result)
    # Save result of testing data with the optimal threshold
    with open('performance.txt', 'w') as file:
        file.write(json.dumps(testing_result))


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
