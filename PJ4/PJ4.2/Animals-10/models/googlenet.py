import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.googlenet import GoogLeNetOutputs
from config_utils import Args
from typing import Dict


class GoogLeNet(nn.Module):
    def __init__(self, args: Args):
        super(GoogLeNet, self).__init__()
        self.args = args
        self.googlenet = models.googlenet()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.googlenet.load_state_dict(state_dict)
        self.googlenet.fc = nn.Linear(1024, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.googlenet(pixels)
        if type(outputs) == GoogLeNetOutputs:
            outputs = outputs.logits
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/googlenet/googlenet-1378be20.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = GoogLeNet(args).to(args.device)

    print(model(**inputs))
