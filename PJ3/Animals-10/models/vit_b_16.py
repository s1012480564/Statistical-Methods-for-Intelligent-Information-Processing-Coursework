import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict


class VisionTransformer(nn.Module):
    def __init__(self, args: Args):
        super(VisionTransformer, self).__init__()
        self.args = args
        self.vit_b_16 = models.vit_b_16()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.vit_b_16.load_state_dict(state_dict)
        self.vit_b_16.heads.head = nn.Linear(768, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.vit_b_16(pixels)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/vit-b-16/vit_b_16-c867db91.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = VisionTransformer(args).to(args.device)

    print(model(**inputs))
