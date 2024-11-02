import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable

def _align_token_labels(
    word_labels,
    word_ids,
    special_tokens_mask,
    offset_mappings,
    ignore_token_id: int= -100,
    aggregation_strategy: str = "first"
):
    batch_size = special_tokens_mask.size(0)
    seq_len = special_tokens_mask.size(1)
    token_labels = torch.ones((batch_size, seq_len), dtype=torch.long) * ignore_token_id

    for seq_idx in range(batch_size):
        for token_idx in range(seq_len):
            if aggregation_strategy == "first":                
                if special_tokens_mask[seq_idx][token_idx] == 0 and offset_mappings[seq_idx][token_idx][0] == 0:
                    token_labels[seq_idx][token_idx] = word_labels[seq_idx][word_ids[seq_idx][token_idx]]
            elif aggregation_strategy == "mean":
                if special_tokens_mask[seq_idx][token_idx] == 0:
                    token_labels[seq_idx][token_idx] = word_labels[seq_idx][word_ids[seq_idx][token_idx]]
            else:
                raise ValueError("Invalid Aggregation Strategy!!!")

    return token_labels

class CoNLL2003Dataset(Dataset):
    def __init__(
        self, dataset: Dataset,
    ):
        self.dataset = dataset
        self.id2classes = {idx:class_name for idx, class_name in enumerate(self.dataset.features["ner_tags"].feature.names)}
        self.classes2id = {class_name:idx for idx, class_name in enumerate(self.dataset.features["ner_tags"].feature.names)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        words = self.dataset[idx]['tokens']
        labels = self.dataset[idx]['ner_tags']
        return (words, labels)

class CoNLL2003DataManager:
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: Callable,
        aggregation_strategy = "first",
        ignore_token_id: int = -100
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.ignore_token_id = ignore_token_id
        self.aggregation_strategy = aggregation_strategy

    def collate_fn(self, batch):
        words, word_labels = zip(*batch)

        tokens_info = self.tokenizer(
            words,
            is_split_into_words = True if isinstance(words[0], list) else False,
            padding=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt"
        )
        input_ids = tokens_info["input_ids"]
        offset_mappings = tokens_info["offset_mapping"]
        special_tokens_mask = tokens_info["special_tokens_mask"]
        word_ids = [tokens_info.word_ids(idx) for idx in range(input_ids.size(0))]

        tokens_info["labels"] = _align_token_labels(
            word_labels=word_labels,
            word_ids=word_ids,
            special_tokens_mask=special_tokens_mask,
            offset_mappings=offset_mappings,
            ignore_token_id=self.ignore_token_id,
            aggregation_strategy=self.aggregation_strategy
        )
        tokens_info["word_ids"] = word_ids
        tokens_info["word_labels"] = word_labels

        return tokens_info

    def get_train_dataloader(
        self, batch_size = 32,
    ):
        return DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 1, pin_memory=True, collate_fn = self.collate_fn
        )

    def get_validation_dataloader(
        self, batch_size = 32,
    ):
        return DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers = 1, pin_memory=True, collate_fn = self.collate_fn
        )

    def get_test_dataloader(
        self, batch_size = 32,
    ):
        return DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers = 1, pin_memory=True, collate_fn = self.collate_fn
        )