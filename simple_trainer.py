import torch.utils.data
import torchmetrics
from torch import optim, nn
import pytorch_lightning as pl

from mydatasets.simple_tag_dataset import SimpleTag


def two_layer_cnn(in_ch, in_dim, width, linear_size=128, num_classes=2):
    reprs = nn.Sequential(
        nn.Conv2d(in_ch, 4*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width, 8*width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*width*(in_dim // 4)*(in_dim // 4), linear_size),
    )
    linear_layer = nn.Linear(linear_size, num_classes)
    model = nn.Sequential()
    model.add_module("reprs", reprs)
    model.add_module("final_layer", linear_layer)
    return model, reprs


class LitSimple(pl.LightningModule):
    def __init__(self, net, num_classes):
        super().__init__()
        self.net = net
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y, _ = batch
        y_hat = self.net(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        # print(f"train_loss: {loss: 0.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.net(x)
        acc = self.accuracy.update(logits, y)
        return acc

    def validation_epoch_end(self, outs):
        self.log('valid_acc', self.accuracy, on_epoch=True, on_step=False)
        print("Valid acc:", self.accuracy.compute())
        return self.accuracy

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outs):
        return self.validation_epoch_end(outs)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    dat = SimpleTag(0)
    net, reprs = two_layer_cnn(3, 32, width=4, num_classes=dat.num_classes)
    train_loader = torch.utils.data.DataLoader(dat.get_train_dataset(), batch_size=32)
    test_loader = torch.utils.data.DataLoader(dat.get_test_dataset(), batch_size=32)

    model = LitSimple(net, dat.num_classes)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
