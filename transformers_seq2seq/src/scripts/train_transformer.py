import torch
import json
import os

from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from dataset.vie_eng import get_data_manager
from transformers import AutoTokenizer
from models.transformers.model import TransformersSeqToSeq
from config import MAIN_DIR, ARTIFACT_DIR
from train.loss import CustomCrossEntropyLoss
from train.train import run_epoch, TrainState, custom_lr_schedule

def main(
    config_file: str = os.path.join(MAIN_DIR, "scripts", "transformer.json"),
    output_dir: str = os.path.join(ARTIFACT_DIR, "transformer")
):
    
    en_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    vie_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")

    with open(config_file, "r") as f:
        configs = json.load(f)
    
    model_args = dict(
        src_vocab_size=vie_tokenizer.vocab_size,
        tgt_vocab_size=en_tokenizer.vocab_size,
        **configs["model_args"]
    )

    training_args = configs["train_args"]

    vie2en_datamanager = get_data_manager()
    train_dataloader, test_dataloader = vie2en_datamanager.get_data_loader(
        batch_size=training_args["batch_size"], test_split=training_args["test_split"]
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformersSeqToSeq(**model_args).to(device)
    optimizer = Adam(model.parameters(), lr=training_args["lr"], betas=training_args["betas"], eps=training_args["eps"])
    criterion = CustomCrossEntropyLoss(
        label_smoothing=0.1,
        ignore_index = en_tokenizer.convert_tokens_to_ids(en_tokenizer.special_tokens_map['pad_token'])
        ).to(device)
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda x: custom_lr_schedule(x, d_model=model_args["d_model"], warm_up=training_args["warm_up"]))

    train_state = TrainState()
    train_loss, val_loss = [], []
    early_stopping = True
    patience = 5

    for epoch_no in range(training_args["epoch_nos"]):
        print(f"Epoch: {epoch_no}")
        model.train()
        _, _, train_step_loss = run_epoch(
            data_iter = train_dataloader,
            model = model,
            loss_compute = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            train_state = train_state,
            accumulation_no = training_args["accumulation_no"],
            mode = "train"
        )
        train_loss.extend(train_step_loss)
        model.eval()
        _, _, val_step_loss = run_epoch(
            data_iter = test_dataloader,
            model = model,
            loss_compute = criterion,
            optimizer = None,
            scheduler = None,
            train_state = train_state,
            accumulation_no = training_args["accumulation_no"],
            mode = "eval"
        )
        val_loss.extend(val_step_loss)
        if early_stopping and train_state.patience_counter > patience:
            break
        
        ckpt_folder = os.path.join(output_dir, "checkpoints", "epoch_{}".format(epoch_no))
        os.makedirs(ckpt_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_folder, "model.pt"))
        
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

if __name__ == "__main__":
    import fire
    fire.Fire(main)
    