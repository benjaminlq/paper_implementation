import torch
import torch.nn as nn

from typing import List, Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from .training import TrainState, custom_lr_schedule
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score

def _extract_first_word_indices(
    word_ids: List[int]
):
    extracted_indices = []
    curr_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id != curr_word_id:
            extracted_indices.append(idx)
            curr_word_id = word_id
    return extracted_indices

def _map_words_to_tokens(
    word_ids: List[int]
):
    words_to_tokens_mapping = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id in words_to_tokens_mapping:
                words_to_tokens_mapping[word_id].append(idx)
            else:
                words_to_tokens_mapping[word_id] = [idx]

    return words_to_tokens_mapping

def word_level_prediction(
    probs: torch.Tensor, # (token_seq_len, num_classes)
    word_ids: List[int], # (word_seq_len)
    aggregation_strategy: str = "first"
):
    if aggregation_strategy == "first":
        extracted_indices = _extract_first_word_indices(word_ids)
        extracted_probs = probs[extracted_indices, :] # (word_seq_len, num_classes)
        return extracted_probs.argmax(dim=-1)
    elif aggregation_strategy == "mean":
        words_to_tokens_mapping = _map_words_to_tokens(word_ids)
        print(words_to_tokens_mapping)
        token_probs = []
        for word_id, token_indices in words_to_tokens_mapping.items():
            extracted_probs = probs[token_indices, :] # (n_tokens_per_word, num_classes)
            avg_probs = extracted_probs.mean(dim=0) # (num_classes)
            token_probs.append(avg_probs)
        return torch.stack(token_probs, dim=0).argmax(dim=-1)
    else:
        raise ValueError("Invalid Aggregation Strategy Type!!!")
    

class BERTTokenClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer_args: Dict,
        data_manager,
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer_args = optimizer_args
        self.device = device
        self.train_state = TrainState()
        self.accumulation_steps = optimizer_args["accumulation_steps"]
        self.data_manager = data_manager

        self.optimizer = AdamW(
            params = self.model.parameters(),
            lr = optimizer_args["lr"],
            betas = optimizer_args["betas"],
            weight_decay = optimizer_args["weight_decay"]
        )

        self.scheduler = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: custom_lr_schedule(
                x, warm_up=optimizer_args["warmup"], total_steps=optimizer_args["total_steps"]
                )
            )
        self.loss_criterion = nn.CrossEntropyLoss(ignore_index=self.data_manager.ignore_token_id)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def run_epoch(
        self,
        epoch: int,
        dataloader: DataLoader,
        train_state: Optional[TrainState] = None,
        is_train: bool = True
    ):
        aggregation_strategy = self.data_manager.aggregation_strategy
        train_state = train_state or self.train_state
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        mode = "Train" if is_train else "Eval"

        data_iter = tqdm(
            enumerate(dataloader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        epoch_loss = 0.0
        epoch_steps = 0
        all_word_preds = []
        all_word_labels = []

        for idx, batch in data_iter:
            input_ids = batch["input_ids"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            word_labels = batch["word_labels"]
            word_ids = batch["word_ids"]

            if is_train:
                outs = self.model(
                    input_ids=input_ids, token_type_ids=token_type_ids, attn_mask=attn_mask,
                )
                logits = outs.view(-1, outs.size(-1))
                labels = labels.view(-1)
                loss = self.loss_criterion(logits, labels)
            else:
                with torch.no_grad():
                    outs = self.model(
                        input_ids=input_ids, token_type_ids=token_type_ids, attn_mask=attn_mask,
                    )
                    logits = outs.view(-1, outs.size(-1))
                    labels = labels.view(-1)
                    loss = self.loss_criterion(logits, labels)

            probs = F.softmax(outs, dim=-1)

            for seq_idx in range(probs.size(0)):
                word_pred = word_level_prediction(
                    probs=probs[seq_idx, :],
                    word_ids=word_ids[seq_idx],
                    aggregation_strategy=aggregation_strategy
                ).detach().cpu()
                word_label = word_labels[seq_idx]

                all_word_preds.extend(word_pred.tolist())
                all_word_labels.extend(word_label)

            if is_train:
                self.train_state.n_steps += 1
                loss.backward()
                if self.train_state.n_steps % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.train_state.n_updates += 1

                lr = self.optimizer.param_groups[0]["lr"]
                self.train_state.lr.append(lr)
                self.train_state.train_loss["step"].append(loss.item())
                self.scheduler.step()

            else:
                self.train_state.val_loss["step"].append(loss.item())

            epoch_loss += loss.item()
            epoch_steps += 1

            if (idx + 1) % 100 == 0:
                avg_loss = epoch_loss / epoch_steps

                print(
                    f"Epoch {epoch} - Step {self.train_state.n_steps}, mode {mode}: avg_loss={avg_loss}")

        avg_loss = epoch_loss / epoch_steps
        epoch_micro_acc = accuracy_score(all_word_labels, all_word_preds)
        epoch_macro_precision = precision_score(all_word_labels, all_word_preds, average="macro", zero_division=np.nan)
        epoch_macro_recall = recall_score(all_word_labels, all_word_preds, average="macro", zero_division=np.nan)

        if is_train:
            train_state.train_loss["epoch"].append(avg_loss)
        else:
            train_state.val_loss["epoch"].append(avg_loss)

        print(
            f"Whole Epoch {epoch}, mode {mode}: avg_loss={avg_loss}, accuracy={epoch_micro_acc}, macro precision={epoch_macro_precision}, macro recall={epoch_macro_recall}")

    def train(
        self,
        training_args: Dict
        ):
        train_dataloader = self.data_manager.get_train_dataloader(batch_size=training_args["batch_size"])
        validation_dataloader = self.data_manager.get_validation_dataloader(batch_size=training_args["batch_size"])

        for epoch_no in range(training_args["n_epochs"]):
            self.run_epoch(
                epoch=epoch_no,
                dataloader=train_dataloader,
                is_train=True
            )
            torch.cuda.empty_cache()
            self.run_epoch(
                epoch=epoch_no,
                dataloader=validation_dataloader,
                is_train=False
            )

    def test(
        self,
        batch_size: int = 32
        ):
        test_dataloader = self.data_manager.get_test_dataloader(batch_size=batch_size)
        self.run_epoch(
            epoch=0,
            dataloader=test_dataloader,
            is_train=False
        )