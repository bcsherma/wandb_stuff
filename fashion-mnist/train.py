import argparse

import pytorch_lightning as pl
import wandb
from torch import utils
from torchvision import datasets, transforms

import model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run = wandb.init(project="fashion-mnist", job_type="model-train", config=args)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    train_set = datasets.FashionMNIST("./data/", download=True, transform=trans)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = utils.data.random_split(train_set, [train_size, val_size])
    train_loader = utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=8
    )
    val_loader = utils.data.DataLoader(
        val_set, batch_size=args.batch_size, num_workers=8
    )
    net = model.FashionMNISTCNN(run.config["learning_rate"])
    logger = pl.loggers.WandbLogger(log_model="all")
    trainer = pl.Trainer(
        logger=logger, max_epochs=10, gpus=1, default_root_dir="./checkpoints/"
    )
    trainer.fit(net, train_loader, val_loader)
    run.finish()
