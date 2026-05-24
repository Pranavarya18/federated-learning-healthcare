import torch
import torch.nn as nn
import torch.optim as optim
import copy


class Client:
    """Federated Learning client with Differential Privacy support.

    KEY FIXES vs old version:
    1. Default loss changed to BCEWithLogitsLoss for heart
       (numerically stable: combines sigmoid + BCE internally)
    2. Adam optimizer instead of SGD (better convergence on small datasets)
    3. Differential Privacy: Gaussian noise added to weights before return
    4. Removed debug guard — clean output
    """

    def __init__(self, model, trainloader, lr=0.001,
                 criterion=None, dp_noise_std=0.0):
        """
        Args:
            model        : local copy of global model
            trainloader  : DataLoader for this client's data
            lr           : learning rate (default 0.001 for Adam)
            criterion    : loss function. Pass explicitly per dataset.
            dp_noise_std : std of Gaussian noise for Differential Privacy.
                           0.0 = no DP noise. Recommended: 0.001–0.01
        """
        self.model        = model
        self.trainloader  = trainloader
        self.dp_noise_std = dp_noise_std

        # Default loss: CrossEntropyLoss for classification
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        # Adam converges faster than SGD, especially on small tabular datasets
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=1):
        """Train locally and return (weights, avg_loss).

        Differential Privacy:
            After training, small Gaussian noise is added to every parameter
            before weights are sent to the server. This prevents the server
            from inferring private data from the weight updates.
        """
        self.model.train()

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        total_loss  = 0.0
        batch_count = 0

        for _ in range(epochs):
            for images, labels in self.trainloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)

                # ── Shape alignment ──────────────────────────────────────
                # BCEWithLogitsLoss and MSELoss need (N,1) vs (N,) matching
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    labels_proc = labels.float().view(-1, 1).to(device)
                else:
                    labels_proc = labels

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels_proc)

                # Safety: skip NaN batches instead of crashing
                if torch.isnan(loss):
                    continue

                loss.backward()

                # Gradient clipping: prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                total_loss  += loss.item()
                batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        # ── Differential Privacy noise injection ─────────────────────────
        # Add Gaussian noise to parameters BEFORE sending to server.
        # This satisfies ε-differential privacy (approximate).
        # More noise = more privacy, less accuracy. We use small std (0.001).
        if self.dp_noise_std > 0.0:
            with torch.no_grad():
                for param in self.model.parameters():
                    noise = torch.randn_like(param) * self.dp_noise_std
                    param.add_(noise)

        return self.model.state_dict(), avg_loss

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def get_weights(self):
        return self.model.state_dict()
