import torch
import torch.nn as nn
import torch.nn.functional as F
from config_utils import Args
from typing import Dict


class LeNet(nn.Module):
    def __init__(self, args: Args):
        super(LeNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        x = F.relu(self.conv1(pixels))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        outputs = self.fc3(x)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/vgg19-bn/vgg19_bn-c79401a0.pth")

    inputs = {"pixels": torch.randn(4, 3, 32, 32, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = LeNet(args).to(args.device)

    print(model(**inputs))
