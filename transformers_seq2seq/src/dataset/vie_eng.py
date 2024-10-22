import torch
import os

from config import DATA_DIR
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable
from transformers import AutoTokenizer
from src.models.transformers.block import get_masked_attention_mask

class VieToEngDataset(Dataset):
    def __init__(
        self,
        vie_sentences: List[str],
        en_sentences: Optional[List[str]] = None,
    ):
        self.vie_sentences = vie_sentences
        self.en_sentences = en_sentences

    def __len__(self):
        return len(self.vie_sentences)

    def __getitem__(self, idx):
        return (self.vie_sentences[idx], self.en_sentences[idx]) if self.en_sentences else self.vie_sentences[idx]

class VieToEngDataManager:
    def __init__(
        self,
        vie_sentences: List[str],
        en_sentences: Optional[List[str]] = None,
        vie_tokenizer: Callable = None,
        en_tokenizer: Optional[Callable] = None,
        max_length: int = 4096,
        seed: int = 42
    ):
        self.dataset = VieToEngDataset(
            vie_sentences=vie_sentences, en_sentences=en_sentences
        )
        self.max_length = max_length
        self.vie_tokenizer = vie_tokenizer or AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
        self.en_tokenizer = en_tokenizer or AutoTokenizer.from_pretrained("bert-large-uncased")
        self.generator = torch.Generator().manual_seed(seed)

    def _collate_fn(
        self, batch_data: List
    ):
        if self.dataset.en_sentences:
            src = [sample[0] for sample in batch_data]
            tgt = [sample[1] for sample in batch_data]
        else:
            src = batch_data
            tgt = None
        tokenized_src = self.vie_tokenizer(
            src, padding=True, truncation=True, return_tensors="pt", max_length = self.max_length
        )
        if tgt:
            tokenized_tgt = self.en_tokenizer(
                tgt, padding=True, truncation=True, return_tensors="pt", max_length = self.max_length
            )
            tgt_input_ids = tokenized_tgt["input_ids"][:, :-1]
            tgt_labels = tokenized_tgt["input_ids"][:, 1:]
            masked_att_mask = get_masked_attention_mask(tgt_input_ids.size(-1))
            tgt_mask = tokenized_tgt["attention_mask"][:, :-1].unsqueeze(-2) & masked_att_mask
            ntokens = tokenized_tgt["attention_mask"][:, 1:].sum()

        return {
            "src_input_ids": tokenized_src["input_ids"],
            "src_mask": tokenized_src["attention_mask"].unsqueeze(-2),
            "tgt_input_ids": tgt_input_ids,
            "tgt_labels": tgt_labels,
            "tgt_mask": tgt_mask,
            "ntokens": ntokens 
        }

    def get_data_loader(
        self, dataset: Optional[Dataset] = None, batch_size: Optional[int] = 32, shuffle: bool = True, test_split: Optional[float] = 0.2
    ):
        dataset = dataset or self.dataset
        if test_split:
            train_dataset, test_dataset = torch.utils.data.random_split(
                self.dataset,
                lengths=[1-test_split, test_split],
                generator=self.generator
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=lambda x: self._collate_fn(x)
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=lambda x: self._collate_fn(x)
            )
            return train_dataloader, test_dataloader

        else:
            return DataLoader(
                dataset,
                batch_size,
                shuffle=shuffle,
                drop_last=False,
                collate_fn=lambda x: self._collate_fn(x)
            )
            
def get_data_manager(
    vie_data_path: str = os.path.join(DATA_DIR, "vi_sents"),
    eng_data_path: str = os.path.join(DATA_DIR, "en_sents")
) -> VieToEngDataManager:

    with open(vie_data_path, "r", encoding="utf-8") as f:
        vie_sentences = f.readlines()

    with open(eng_data_path, "r", encoding="utf-8") as f:
        eng_sentences = f.readlines()

    en_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    vie_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    vie2eng_datamanager = VieToEngDataManager(
        vie_sentences = vie_sentences,
        en_sentences = eng_sentences,
        vie_tokenizer = vie_tokenizer,
        en_tokenizer = en_tokenizer
    )
    
    return vie2eng_datamanager