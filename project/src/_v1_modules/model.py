"""
model.py
========
Phase 4a - the deep network itself. Architecture only: no training, no
inference loop, no uncertainty logic.

The dropout layers are not just regularisation. They are the mechanism the
whole paper depends on: mc_dropout.py keeps them active at inference time to
turn one deterministic network into a distribution over predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class UADNN(nn.Module):
    """
    21 -> 128 -> 64 -> 32 -> 1 fully connected classifier.

    BatchNorm after the first two hidden layers, dropout after all three,
    sigmoid on the output (so the forward pass returns a probability, not a
    logit - every downstream explainer assumes this).
    """

    def __init__(self, input_dim: int = config.N_FEATURES):
        super().__init__()
        h1, h2, h3 = config.HIDDEN_UNITS
        p1, p2, p3 = config.DROPOUT_RATES

        self.fc1 = nn.Linear(input_dim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.drop1 = nn.Dropout(p1)

        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.drop2 = nn.Dropout(p2)

        self.fc3 = nn.Linear(h2, h3)
        self.drop3 = nn.Dropout(p3)

        self.fc4 = nn.Linear(h3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.fc3(x)))
        return torch.sigmoid(self.fc4(x))


def build_model(input_dim: int = config.N_FEATURES) -> UADNN:
    """Create a freshly initialised network under the configured seed."""
    config.set_seeds()
    return UADNN(input_dim=input_dim)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (reported in the paper's model table)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model: nn.Module) -> str:
    """Layer-by-layer parameter counts, for the methods section."""
    lines = [f"{'layer':<20}{'parameters':>12}"]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            lines.append(f"{name:<20}{parameter.numel():>12,}")
    lines.append(f"{'TOTAL':<20}{count_parameters(model):>12,}")
    return "\n".join(lines)


# =====================================================================
# Checklist - what this file does
# ---------------------------------------------------------------------
# [x] Defines UADNN: 21 -> 128 -> 64 -> 32 -> 1, ReLU activations.
# [x] BatchNorm1d after hidden layers 1 and 2.
# [x] Dropout after all three hidden layers (0.3, 0.3, 0.2) - kept active
#     at inference time by mc_dropout.py to produce epistemic uncertainty.
# [x] Sigmoid output, so forward() returns P(diabetes) directly; training
#     therefore uses BCELoss, not BCEWithLogitsLoss.
# [x] build_model() seeds first so two runs give the same initialisation.
# [x] count_parameters() / describe() report model size for the paper.
# [x] Contains no training or inference code on purpose.
# =====================================================================
