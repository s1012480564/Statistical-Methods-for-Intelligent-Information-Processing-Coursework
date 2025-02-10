import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict


class DenseNet(nn.Module):
    def __init__(self, args: Args):
        super(DenseNet, self).__init__()
        self.args = args
        self.densenet201 = models.densenet201()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        state_dict_corrected = self._bug_free(state_dict)
        self.densenet201.load_state_dict(state_dict_corrected)
        self.densenet201.classifier = nn.Linear(1920, args.num_classes)

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.densenet201(pixels)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}

    @staticmethod
    def _bug_free(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # densenet 保存的参数键名有 bug，服了
        state_dict_corrected = {}
        for key in state_dict.keys():
            key_corrected = key.replace(".1", "1")
            key_corrected = key_corrected.replace(".2", "2")
            state_dict_corrected[key_corrected] = state_dict[key]
        return state_dict_corrected


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/densenet201/densenet201-c1103571.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = DenseNet(args).to(args.device)

    print(model(**inputs))
