import argparse
import os

import lightning.pytorch as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader
from transformers import BigBirdConfig, BigBirdForMaskedLM, PreTrainedTokenizerFast

from codon_transformer_2.collators import MaskedTokenizerCollator

NUM_ORGANISMS = 4742
LEARNING_RATE = 1e-4
WARMUP_FRACTION = 0.1
MAX_EPOCHS = 10
BATCH_SIZE = 128
NUM_WORKERS = 16
DEBUG = True


class TrainHarness(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
        )
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=LEARNING_RATE,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=WARMUP_FRACTION,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        self.log_dict(
            dictionary={
                "loss": outputs.loss,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            },
            on_step=True,
            prog_bar=True,
        )
        return outputs.loss


class EpochCheckpoint(pl.Callback):
    def __init__(self, save_interval):
        super().__init__()
        self.checkpoint_dir = "."
        self.save_interval = save_interval

    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch % self.save_interval == 0 or current_epoch == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch}.ckpt")
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}\n")


def main(args):
    pl.seed_everything(123)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Load the tokenizer and model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="data/codon_transformer_tokenizer.json",
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        max_len=1024,
    )
    config = BigBirdConfig(
        vocab_size=len(tokenizer),
        type_vocab_size=NUM_ORGANISMS,
        sep_token_id=tokenizer.sep_token_id,
    )
    model = BigBirdForMaskedLM(config=config)
    # self.model.bert.set_attention_type("block_sparse")
    model.bert.set_attention_type("original_full")  # <=== Address this.
    harnessed_model = TrainHarness(model)

    # Load the training data
    train_data = wds.WebDataset(
        "/path/to/Codons/webdataset/shard-{000000..003855}.tar.gz",
        nodesplitter=wds.shardlists.split_by_node,
        workersplitter=wds.shardlists.split_by_worker,
    )
    data_loader = DataLoader(
        dataset=train_data,
        collate_fn=MaskedTokenizerCollator(tokenizer),
        batch_size=12,
        num_workers=4,
        persistent_workers=True,
    )

    # Setup trainer and callbacks
    # save_checkpoint = EpochCheckpoint(args.checkpoint_dir, args.save_interval)
    trainer = pl.Trainer(
        default_root_dir=".",
        strategy="deepspeed",
        accelerator="gpu",
        devices=4,
        precision="bf16-mixed",
        max_epochs=1,
        deterministic=False,
        enable_checkpointing=True,
        limit_train_batches=2_632_363,
        # callbacks=[save_checkpoint],
        # accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Pretrain the model
    trainer.fit(harnessed_model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training harness.")
    # parser.add_argument(
    #     "--train_data_path",
    #     type=str,
    #     required=True,
    #     help="Path to the training data JSON file",
    # )
    args = parser.parse_args()
    main(args)
