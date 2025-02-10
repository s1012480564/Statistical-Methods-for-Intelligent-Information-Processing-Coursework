import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict


class VGG(nn.Module):
    def __init__(self, args: Args):
        super(VGG, self).__init__()
        self.args = args
        self.vgg19_bn = models.vgg19_bn()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.vgg19_bn.load_state_dict(state_dict)
        self.vgg19_bn.classifier[-1] = nn.Linear(4096, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.vgg19_bn(pixels)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/vgg19-bn/vgg19_bn-c79401a0.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = VGG(args).to(args.device)

    print(model(**inputs))
