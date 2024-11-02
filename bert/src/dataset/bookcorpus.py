import torch
import torch.nn as nn
import random

from tqdm import tqdm
from typing import List, Optional, Callable
from torch.utils.data import Dataset

class BookCorpusPretrainDataset(Dataset):
    def __init__(
        self,
        corpus: List[str],
        tokenizer: Callable,
        corpus_len: Optional[List[int]] = None,
        seq_len: int = 128,
        prob: float = 0.15,
        mask_prob: float = 0.80,
        random_prob: float = 0.10,
        **kwargs
    ):
        self.corpus = corpus
        self.corpus_len = corpus_len
        if not self.corpus_len:
            batch_size = 256
            self.corpus_len = []

            for start_idx in tqdm(range(0, len(self.corpus), batch_size)):
                batch_data = tokenizer(self.corpus[start_idx:start_idx+batch_size]["text"], add_special_tokens=False)["input_ids"]
                for seq in batch_data:
                    self.corpus_len.append(len(seq))

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size
        self.special_token_map = {}
        for token_type, token_str in self.tokenizer.special_tokens_map.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            self.special_token_map[token_type] = (token_str, token_id)

        self.non_special_token_ids = [token_id for token_id in range(self.vocab_size) if token_id not in self.tokenizer.all_special_ids]
        self.prob = prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob

    def get_text_span(self, idx, max_len: int):
        sentences = []
        token_counts = 0
        if self.corpus_len[idx] > max_len:
            return sentences

        sentences.append(self.corpus[idx])
        token_counts += self.corpus_len[idx]

        curr_idx = idx
        while token_counts < max_len and curr_idx < (len(self) - 1):
            next_sentence = self.corpus[curr_idx + 1]
            curr_idx += 1
            if token_counts + self.corpus_len[curr_idx] <= max_len:
                sentences.append(next_sentence)
                token_counts += self.corpus_len[curr_idx]
            else:
                break

        return sentences

    def get_seq_len(self, input_str):
        return len(self.tokenizer.encode(input_str, add_special_tokens=False))

    def __len__(self):
        return len(self.corpus)

    def __sample_mask(
        self,
        input_ids: List[int],
        prob: Optional[float] = None,
        mask_prob: Optional[float] = None,
        random_prob: Optional[float] = None,
    ):
        prob = prob or self.prob
        mask_prob = mask_prob or self.mask_prob
        random_prob = random_prob or self.random_prob

        masked_seq = []
        masked_labels = []
        for token_id in input_ids:
            if token_id not in self.tokenizer.all_special_ids:
                if random.random() < 0.15:
                    prob = random.random()
                    if prob < mask_prob: # Change to MASK token (80%)
                        masked_seq.append(self.special_token_map["mask_token"][1])
                    elif prob < (mask_prob + random_prob): # Change to a random token (10%)
                        masked_seq.append(random.choice(self.non_special_token_ids))
                    else: # Does not change (10%)
                        masked_seq.append(token_id)
                    masked_labels.append(token_id)
                else:
                    masked_seq.append(token_id)
                    masked_labels.append(self.special_token_map["pad_token"][1])
            else:
                masked_seq.append(token_id)
                masked_labels.append(self.special_token_map["pad_token"][1])
        return (masked_seq, masked_labels)

    def __getitem__(self, idx):
        if idx == (len(self) - 1):
            idx = random.randint(0, len(self) - 2)
        while True:
            sentences = self.get_text_span(idx, max_len=self.seq_len - 3)
            if len(sentences) >= 2:
                break
            else:
                idx = random.randint(0, len(self) - 2)

        max_len = random.randint(1, len(sentences) - 1)

        split_threshold = random.randint(1, max_len)
        nsp_prob = random.random()
        sent_A = " ".join(sentences[:split_threshold])
        if nsp_prob > 0.5:
            sent_B = " ".join(sentences[split_threshold:])
            nsp_label = 0
        else: # Need to sample a random span
            while True:
                random_span = self.get_text_span(random.randint(0, len(self)-1), max_len=self.seq_len - self.get_seq_len(sent_A) - 3)
                if len(random_span) >= 2:
                    break
                else:
                    split_threshold -= 1
                    sent_A = " ".join(sentences[:split_threshold])

            random_split_threshold = random.randint(1, len(random_span) - 1)
            sent_B = " ".join(random_span[random_split_threshold:])
            nsp_label = 1

        tokens_info = self.tokenizer(
            sent_A, sent_B, truncation=False, padding='max_length', max_length=self.seq_len
        )
        input_ids = tokens_info["input_ids"]
        attn_mask = tokens_info["attention_mask"]
        token_type_ids = tokens_info["token_type_ids"]

        input_ids, labels = self.__sample_mask(input_ids=input_ids)
        return {
            "input_ids": torch.tensor(input_ids),
            "attn_mask": torch.tensor(attn_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "mlm_label": torch.tensor(labels),
            "nsp_label": torch.tensor([nsp_label]),
        }