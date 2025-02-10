import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict
from torch import Tensor
import torch.nn.functional as F
from torch.nn import MultiheadAttention


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


# 复现 GLIP 思想，两个 backbone vis-encode 结果交叉注意力残差然后等大线性层。然后由于任务不同，这里就直接拼接后 MLP 了
class GLIP(nn.Module):
    def __init__(self, args: Args):
        super(GLIP, self).__init__()
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

        self.mha1 = MultiheadAttention(embed_dim=768, num_heads=12, kdim=1920, vdim=1920, batch_first=True)
        self.mha2 = MultiheadAttention(embed_dim=1920, num_heads=12, kdim=768, vdim=768, batch_first=True)

        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(1920, 1920)

        self.mlp = nn.Sequential(
            nn.Linear(1920 + 768, 512),
            nn.GELU(),
            nn.Linear(512, args.num_classes)
        )

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        vit_encode = self.vit_b_16(pixels)
        densenet_encode = self.densenet201(pixels)

        out1 = vit_encode + self.mha1(vit_encode, densenet_encode, densenet_encode)[0]
        out1 = self.linear1(out1)

        out2 = densenet_encode + self.mha2(densenet_encode, vit_encode, vit_encode)[0]
        out2 = self.linear2(out2)

        outputs = self.mlp(torch.cat((out1, out2), dim=-1))
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

    model = GLIP(args).to(args.device)

    print(model(**inputs))
