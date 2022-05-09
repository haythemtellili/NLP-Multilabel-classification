import json
import logging
import os
import sys
import numpy as np

import transformers
import torch

from model import MultilabelClassifier
from utils import LabelEncoder
import config as config

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 90  # this is the max length of the sentence

print("Loading tokenizer...")
tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-multilingual-cased"
)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder.load(
        os.path.join(model_dir, "label_encoder.json")
    )
    model = MultilabelClassifier(len(label_encoder))
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "best_model.pt"),
            map_location=torch.device(device),
        )
    )
    model.to(device)
    model.eval()
    return model, label_encoder


def input_fn(request_body, content_type="application/json"):
    logger.info("Deserializing the input data.")
    if content_type == "application/json":
        data = json.loads(request_body)
        print("================ input sentences ===============")
        print(data)
        url = data["url"]
        url = " ".join(url.split())
        inputs = tokenizer.encode_plus(
            url,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        return ids, mask


def predict_fn(input_data, model_artifacts):
    model, label_encoder = model_artifacts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ids, mask = input_data
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    with torch.no_grad():
        outputs = model(ids=ids, mask=mask)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        result = np.array([np.where(prob >= config.THRESHOLD, 1, 0) for prob in outputs])
        result = str(label_encoder.decode([result[0]])[0])
        print("=============== inference result =================")
        print(result)

    return result
