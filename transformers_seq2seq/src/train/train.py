import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class TrainState:
    step: int = 0
    n_updates: int = 0
    samples: int = 0
    tokens: int = 0
    best_loss: float = float("inf")
    patience_counter: int = 0
    
def run_epoch(
    data_iter: DataLoader,
    model: nn.Module,
    loss_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_state,
    accumulation_no: int=1,
    mode: str="train", # train/eval
):
    total_tokens = 0
    total_loss = 0.0
    n_updates = 0
    all_losses = []

    device = next(model.parameters()).device

    for i, batch in tqdm(enumerate(data_iter), total=len(data_iter)):
        src = batch["src_input_ids"].to(device)
        tgt = batch["tgt_input_ids"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)
        ntokens = batch["ntokens"]
        labels = batch["tgt_labels"].to(device)
        
        probs = model(
            src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask
        )
        
        batch_loss = loss_criterion(
            probs.contiguous().view(-1, probs.size(-1)),
            labels.contiguous().view(-1)
        )
        
        loss = batch_loss / ntokens

        if mode == "train":
            loss.backward()

            if i % accumulation_no == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_updates += 1
                train_state.n_updates += 1
            
            scheduler.step()

            train_state.step += 1
            train_state.samples += src.size(0)
            train_state.tokens += ntokens

        all_losses.append(loss.item())
        total_loss += batch_loss.item()
        total_tokens += ntokens

        if i % 200 == 1 and mode == "train":
            lr = optimizer.param_groups[0]["lr"]
            print(
                "Epoch Step: {:6d} | Gradient Update Step: {:3d} | Loss Per Token: {:.5f} | Learning Rate: {:6.1e}".format(
                    i, n_updates, loss, lr)
                )
            
    if mode == "eval":
        print("Total per token eval loss:", total_loss / total_tokens)
        if total_loss / total_tokens < train_state.best_loss:
            train_state.best_loss = total_loss / total_tokens
            train_state.patience_counter = 0
        else:
            train_state.patience_counter += 1
            
    return total_loss / total_tokens, train_state, all_losses

def custom_lr_schedule(
    step_no: int,
    d_model: int = 512,
    warm_up: int = 4000,      
) -> float:
    if step_no == 0:
        step_no = 1
    return d_model ** (-0.5) * min(step_no**(-0.5), step_no*(warm_up**(-1.5)))