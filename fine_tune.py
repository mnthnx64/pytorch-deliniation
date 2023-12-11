import lightning as L
from models.unet_models import UNetResNet50_9
import torch


class SegmentationModel(L.LightningModule):
    def __init__(self, num_classes=1):
        super(SegmentationModel, self).__init__()
        self.model = UNetResNet50_9(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch['image']
        filled = batch['filled']
        border = batch['border']
        filled_pred, border_pred = self.model(images)
        filled_loss = self.criterion(filled_pred, filled)
        border_loss = self.criterion(border_pred, border)
        loss = filled_loss + border_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        filled = batch['filled']
        border = batch['border']
        filled_pred, border_pred = self.model(images)
        filled_loss = self.criterion(filled_pred, filled)
        border_loss = self.criterion(border_pred, border)
        loss = filled_loss + border_loss
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch['image']
        filled = batch['filled']
        border = batch['border']
        filled_pred, border_pred = self.model(images)
        filled_loss = self.criterion(filled_pred, filled)
        border_loss = self.criterion(border_pred, border)
        loss = filled_loss + border_loss
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def criterion(self, outputs, targets):
        # Convert outputs and targets to binary predictions
        predictions = torch.round(torch.sigmoid(outputs))
        targets = targets.float()

        # Calculate true positives, false positives, and false negatives
        true_positives = torch.sum(predictions * targets)
        false_positives = torch.sum(predictions) - true_positives
        false_negatives = torch.sum(targets) - true_positives

        # Calculate precision, recall, and F1 score
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        