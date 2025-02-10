import torch
import torch.nn as nn
from torchvision import models
from config_utils import Args
from typing import Dict
from torch import Tensor
import torch.nn.functional as F


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


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


# 复现 MVC
class MVC(nn.Module):
    def __init__(self, args: Args):
        super(MVC, self).__init__()
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

        self.linear_vit = nn.Linear(768, args.num_classes)
        self.linear_densenet = nn.Linear(1920, args.num_classes)

        self.global_step = 0

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.args.num_classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.args.num_classes, 1), b[1].view(-1, 1, self.args.num_classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.args.num_classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def forward(self, pixels: torch.Tensor, labels: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        vit_encode = self.vit_b_16(pixels)
        densenet_encode = self.densenet201(pixels)
        evidence = [self.linear_vit(vit_encode), self.linear_densenet(densenet_encode)]
        loss = 0
        alpha = dict()
        for v_num in range(2):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(labels, alpha[v_num], self.args.num_classes, self.global_step, 113)  # 原文是前 10% epoch 参数线性增长到 1，这里我就手动固定前 10 % steps
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(labels, alpha_a, self.args.num_classes, self.global_step, 113)
        self.global_step += 1
        loss = torch.mean(loss)
        return {"loss": loss, "logits": evidence_a}

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

    model = MVC(args).to(args.device)

    print(model(**inputs))
