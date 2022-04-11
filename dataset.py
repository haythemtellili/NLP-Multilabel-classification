import config
import torch


class MultilabelDataset:
    def __init__(self, urls, labels):
        self.urls = urls
        self.labels = labels
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, item):
        url = str(self.urls[item])
        url = " ".join(url.split())

        inputs = self.tokenizer.encode_plus(
            url,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[item], dtype=torch.float),
        }
