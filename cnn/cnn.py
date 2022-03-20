import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNN(pl.LightningModule):

    def __init__(self, 
        num_classes: int,
        activation: nn.Module = nn.ReLU(), 
        probability: float = 0.2
        ):
        super().__init__()
        self.save_hyperparameters()

        # TODO: Split this up so a hyperparamter search can be done

        self.model = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(5, 5)),
            activation,
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(30, 15, kernel_size=(3, 3)),
            activation,
            nn.MaxPool2d((2, 2)),
            nn.Dropout(probability),
            nn.Flatten(),
            nn.Linear(375, 128),
            activation,
            nn.Linear(128, 50),
            activation,
            nn.Linear(50, num_classes),
            activation,
            nn.Softmax()
        )

    def forward(self, x):
        probabilities = self.model(x)
        return probabilities

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
