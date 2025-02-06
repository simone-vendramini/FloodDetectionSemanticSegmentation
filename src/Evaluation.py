import torch
import numpy as np

from torchmetrics.segmentation import DiceScore


class Evaluation:
    def __init__(self, model, dataloader, device, fine_tune=False):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.fine_tune = fine_tune

    def evaluate_model(self):
        self.model.eval()
        accuracy, dice_score, mean_iou = [], [], []
        class_occurrences = {i: 0 for i in range(10)}

        with torch.no_grad():
            for _, (X, y) in enumerate(self.dataloader):
                X, y = X.to(self.device), y.to(self.device)

                for i in range(10):
                    class_occurrences[i] += (y == i).any(dim=(1, 2)).sum().item()
                if self.fine_tune:
                    pred = self.model(X)['out']
                else:
                    pred = self.model(X)
                    
                pred = pred.argmax(dim=1)

                for i in range(len(pred)):
                    accuracy.append(self._pixel_accuracy(pred[i], y[i]))
                    dice_score.append(self._dice_score(pred[i], y[i]))
                    mean_iou.append(self._mean_iou(pred[i], y[i]))

        print("\nClass occurrences in test set:")
        for cls, count in class_occurrences.items():
            print(f"Class {cls}: {count} images")

        return (
                torch.tensor(accuracy).cpu().numpy().mean(),
                torch.tensor(dice_score).cpu().numpy().mean(),
                torch.tensor(mean_iou).cpu().numpy().mean(),
            )

    def _pixel_accuracy(self, pred, label):
        correct_pixels = (pred == label).sum().item()
        total_pixels = label.numel()
        accuracy = correct_pixels / total_pixels
        return accuracy

    def _dice_score(self, pred, label):
        dice_score = DiceScore(
            num_classes=10,
            average="weighted",
            input_format="index",
        )
        return dice_score(pred.unsqueeze(0), label)

    def _mean_iou(self, pred, label):
        intersection = torch.zeros(10)
        union = torch.zeros(10)

        for i in range(10):
            pred_i = pred == i
            label_i = label == i

            if label_i.any():
                intersection[i] = (pred_i & label_i).sum().float()
                union[i] = (pred_i | label_i).sum().float()

        class_present = union > 0
        iou = torch.zeros(10)
        iou[class_present] = intersection[class_present] / union[class_present]

        mean_iou = iou[class_present].mean()

        return mean_iou.item()
