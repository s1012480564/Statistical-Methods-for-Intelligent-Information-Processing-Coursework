import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict
from torch import Tensor
import torch.nn.functional as F


def forward_vit_encode(self, x: torch.Tensor):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)

    # Classifier "token" as used by standard language architectures
    x = x[:, 0]

    # x = self.heads(x)

    return x


def forward_densenet_encode(self, x: Tensor) -> Tensor:
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    # out = self.classifier(out)
    return out


# 复现 PointFusion 思想，两个 backbone vis-encode 结果拼接，然后 MLP
class PointFusion(nn.Module):
    def __init__(self, args: Args):
        super(PointFusion, self).__init__()
        self.args = args

        self.vit_b_16 = models.vit_b_16()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.vit_b_16.load_state_dict(state_dict)
        self.vit_b_16.forward = forward_vit_encode.__get__(self.vit_b_16)

        self.densenet201 = models.densenet201()
        state_dict = torch.load(args.pretrained_path2, weights_only=True)
        state_dict_corrected = self._bug_free(state_dict)
        self.densenet201.load_state_dict(state_dict_corrected)
        self.densenet201.forward = forward_densenet_encode.__get__(self.densenet201)

        self.mlp = nn.Sequential(
            nn.Linear(1920 + 768, 512),
            nn.GELU(),
            nn.Linear(512, args.num_classes)
        )

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        vit_encode = self.vit_b_16(pixels)
        densenet_encode = self.densenet201(pixels)
        outputs = self.mlp(torch.cat((vit_encode, densenet_encode), dim=-1))
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
                pretrained_path="../../../pretrained/vit-b-16/vit_b_16-c867db91.pth",
                pretrained_path2="../../../pretrained/densenet201/densenet201-c1103571.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = PointFusion(args).to(args.device)

    print(model(**inputs))
