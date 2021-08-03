import argparse

import pytorch_lightning as pl
import wandb
from torch import utils
from torchvision import datasets, transforms

import model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run = wandb.init(project="fashion-mnist", job_type="evaluation")
    checkpoint = run.use_artifact(args.model)
    artifact_dir = checkpoint.download(root="./models/")
    net = model.FashionMNISTCNN.load_from_checkpoint(artifact_dir + "/model.ckpt")
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    test_set = datasets.FashionMNIST("./data/", download=True, transform=trans, train=0)
    test_loader = utils.data.DataLoader(test_set, batch_size=512, num_workers=8)
    test_accuracy = pl.metrics.Accuracy()
    trainer = pl.Trainer(gpus=1)
    trainer.test(net, test_loader)
    run.log({"test data": net.test_table})
    run.finish()
