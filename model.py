import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  MNIST MODEL  (unchanged — working)
# ─────────────────────────────────────────────

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ─────────────────────────────────────────────
#  DIABETES MODEL
# ─────────────────────────────────────────────

class DiabetesModel(nn.Module):
    """Regression model for diabetes dataset.

    Uses LayerNorm (safe for any batch size, including batch=1).
    Output is raw scalar — use MSELoss.
    """
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ─────────────────────────────────────────────
#  HEART DISEASE MODEL
# ─────────────────────────────────────────────

class HeartDiseaseModel(nn.Module):
    """Binary classifier for Heart Disease dataset.

    KEY FIXES:
    1. Output = raw LOGIT (no sigmoid inside forward)
       → Use BCEWithLogitsLoss in client (numerically stable)
       → At inference: apply torch.sigmoid() manually
    2. LayerNorm instead of BatchNorm1d
       → BatchNorm crashes when batch_size=1 (last small batch)
       → LayerNorm works for any batch size
    3. Dropout for regularisation on small dataset
    """
    def __init__(self):
        super(HeartDiseaseModel, self).__init__()
        self.fc1   = nn.Linear(13, 64)
        self.ln1   = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(64, 32)
        self.ln2   = nn.LayerNorm(32)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(32, 1)   # raw logit output

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop1(F.relu(self.ln1(self.fc1(x))))
        x = self.drop2(F.relu(self.ln2(self.fc2(x))))
        return self.fc3(x)
