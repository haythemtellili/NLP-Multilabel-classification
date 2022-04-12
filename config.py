import transformers
import torch

PATH_DATA = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
EPOCHS = 30
MAX_LEN = 90
DISTILBERT_PATH = "distilbert-base-multilingual-cased"
TOKENIZER = transformers.DistilBertTokenizer.from_pretrained(
    "distilbert-base-multilingual-cased"
)
MODEL_PATH = "model/best_model.ptbest_model.pt"
LABEL_ENCODER_PATH = "model/best_model.ptlabel_encoder.json"
THRESHOLD = 0.29
