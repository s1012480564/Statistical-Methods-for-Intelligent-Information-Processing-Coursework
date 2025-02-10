import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict
from functools import partial


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


def forward_with_decision(self, input: torch.Tensor, decision: nn.Module):
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    x, _ = self.self_attention(x, x, x, need_weights=False)
    x = self.dropout(x)
    x = x + input

    y = self.ln_2(x)
    y = self.mlp(y)

    out = x + y
    decision_out = decision(out[:, 0])
    decision_out = decision_out.unsqueeze(-1).expand((decision_out.shape[0], decision_out.shape[1], out.shape[-1]))
    out = torch.cat((out, decision_out), dim=1)

    return out


# 复现 Decision Fusion ，每个 block GAP 后 make decision, expand，然后拼回去。
# 对 vit_b_16 那么就每层 [CLS] 处输出 token 然后 MLP 视作 GAP 结果（bert pooler_output 思想）
# 按 bert pooler_output 思想，CLS 处和其他位置区分，MLP 激活使用和 RELU 类不一样的 Tanh，中间层特征和输入特征数量相等
# 分类头不涉及对 CLS 以外部分使用，沿用前面实验表现很好的方案，中间层512，GELU激活
class DecisionFusion(nn.Module):
    def __init__(self, args: Args):
        super(DecisionFusion, self).__init__()
        self.args = args

        self.decisions = nn.Sequential()
        for _ in range(12):
            self.decisions.append(nn.Sequential(
                nn.Linear(768, 768),
                nn.Tanh(),
                nn.Linear(768, args.num_classes)
            ))

        self.vit_b_16 = models.vit_b_16()
        state_dict = torch.load(args.pretrained_path, weights_only=True)
        self.vit_b_16.load_state_dict(state_dict)
        self.vit_b_16.forward = forward_vit_encode.__get__(self.vit_b_16)
        for i in range(12):
            self.vit_b_16.encoder.layers[i].forward = partial(forward_with_decision.__get__(
                self.vit_b_16.encoder.layers[i]), decision=self.decisions[i])

        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, args.num_classes)
        )

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.vit_b_16(pixels)
        outputs = self.classifier(outputs)
        loss = self.args.criterion(outputs, labels.flatten())
        return {"loss": loss, "logits": outputs}


def test_unit():
    from torch.nn import CrossEntropyLoss

    args = Args(device=0, num_classes=6, dropout=0.1, criterion=CrossEntropyLoss(),
                pretrained_path="../../../pretrained/vit-b-16/vit_b_16-c867db91.pth",
                pretrained_path2="../../../pretrained/densenet201/densenet201-c1103571.pth")

    inputs = {"pixels": torch.randn(4, 3, 224, 224, device=args.device),
              "labels": torch.randint(0, 6, (4,), device=args.device)}

    model = DecisionFusion(args).to(args.device)

    print(model(**inputs))
