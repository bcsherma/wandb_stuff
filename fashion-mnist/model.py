import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.nn import functional as F
import wandb

LABELS = [
    "T_SHIRT",
    "PANT",
    "PULLOVER",
    "DRESS",
    "COAT",
    "SANDAL",
    "SHIRT",
    "SNEAKER",
    "BAG",
    "ANKLE_BOOT",
]


class FashionMNISTCNN(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super(FashionMNISTCNN, self).__init__()
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.lin1 = nn.Linear(9 * 7 * 7, 128)
        self.lin2 = nn.Linear(128, 10)
        self.loss = nn.CrossEntropyLoss()
        self.val_loss = nn.CrossEntropyLoss()
        self.test_loss = nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def reset_test_table(self):
        columns = ["image", "label", "prediction"]
        self.test_table = wandb.Table(columns=columns)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        x = nn.Dropout(0.25)(x)
        x = self.lin2(x)
        return x

    def training_step(self, batch, batch_no):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        self.log("train/loss", loss, on_epoch=True)
        preds = torch.argmax(outputs, 1)
        self.train_accuracy(preds, labels)
        self.log("train/accuracy", self.train_accuracy, on_epoch=1, prog_bar=1)
        return loss

    def validation_step(self, batch, batch_no):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.val_loss(outputs, labels)
        self.log("val/loss", loss, on_epoch=True)
        preds = torch.argmax(outputs, 1)
        self.val_accuracy(preds, labels)
        self.log("val/accuracy", self.val_accuracy, on_epoch=1)

    def on_test_epoch_start(self):
        self.reset_test_table()

    def test_step(self, batch, batch_no):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.test_loss(outputs, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        preds = torch.argmax(outputs, 1)
        self.test_accuracy(preds, labels)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=1)
        for im, label, pred in zip(inputs, labels, preds):
            self.test_table.add_data(
                wandb.Image(im), LABELS[int(label)], LABELS[int(pred)]
            )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
