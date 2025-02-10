import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict


class AlexNet(nn.Module):
    def __init__(self, args: Args):
        super(AlexNet, self).__init__()
        self.args = args
        self.alexnet = models.alexnet()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.alexnet.load_state_dict(state_dict)
        self.alexnet.classifier[-1] = nn.Linear(4096, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.alexnet(pixels)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/alexnet/alexnet-owt-7be5be79.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = AlexNet(args).to(args.device)

    print(model(**inputs))
