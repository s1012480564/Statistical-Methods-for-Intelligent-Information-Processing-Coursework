import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict


class ResNet(nn.Module):
    def __init__(self, args: Args):
        super(ResNet, self).__init__()
        self.args = args
        self.resnet152 = models.resnet152()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.resnet152.load_state_dict(state_dict)
        self.resnet152.fc = nn.Linear(2048, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.resnet152(pixels)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from config_utils import Args
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/resnet152/resnet152-394f9c45.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = ResNet(args).to(args.device)

    print(model(**inputs))
