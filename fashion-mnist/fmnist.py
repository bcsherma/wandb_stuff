import wandb
import torch
from torchvision import datasets, transforms
from torch import nn, optim, utils
import pytorch_lightning as pl


class FashionMNISTCNN(pl.LightningModule):
    def __init__(self):

        super(FashionMNISTCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(nn.Linear(4 * 7 * 7, 10))
        self.loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.validation_table = None
        self.reset_validation_table()

    def reset_validation_table(self):
        columns = ["image", "label", "prediction"]
        self.validation_table = wandb.Table(columns=columns)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    def training_step(self, batch, batch_no):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        self.log("train/loss", loss, on_epoch=True)
        preds = torch.argmax(outputs, 1)
        self.train_accuracy(preds, labels)
        self.log("train/accuracy", self.train_accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_no):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.argmax(outputs, 1)
        self.val_accuracy(preds, labels)
        self.log("val/accuracy", self.val_accuracy, on_epoch=True, prog_bar=True)
        for im, pred, label in zip(inputs, preds, labels):
            self.validation_table.add_data(wandb.Image(im), int(pred), int(label))

    def on_validation_epoch_end(self):
        wandb.log({"validation data": self.validation_table})
        self.reset_validation_table()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    wandb.init(project="fashion-mnist")
    train_set = datasets.FashionMNIST(
        "./data/",
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    )
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, validation_set = utils.data.random_split(
        train_set, [train_size, val_size]
    )
    train_loader = utils.data.DataLoader(train_set, batch_size=32)
    val_loader = utils.data.DataLoader(validation_set, batch_size=32)
    model = FashionMNISTCNN()
    trainer = pl.Trainer(logger=pl.loggers.WandbLogger(log_model="all"), max_epochs=2)
    trainer.fit(model, train_loader, val_loader)
