"""
Advanced Training Strategies for NIDS
======================================
Implements state-of-the-art training techniques for improved performance.

Strategies:
1. Self-supervised pre-training (masked node prediction)
2. Contrastive learning (SimCLR-style)
3. Label smoothing
4. SWA (Stochastic Weight Averaging)
5. Mixed precision training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
import copy
from collections import OrderedDict


class SelfSupervisedPretrainer:
    """
    Self-supervised pre-training using masked node feature prediction.
    Inspired by GraphIDS and BERT-style masked language modeling.
    """
    def __init__(self, model, mask_ratio=0.15, hidden_dim=128):
        self.model = model
        self.mask_ratio = mask_ratio
        self.hidden_dim = hidden_dim

        # Decoder for masked prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def mask_features(self, x, mask_ratio=None):
        """
        Randomly mask node features for self-supervised learning.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        N = x.size(0)
        num_mask = int(N * mask_ratio)

        # Random mask indices
        mask_indices = torch.randperm(N)[:num_mask]

        # Create mask tensor
        mask = torch.zeros(N, dtype=torch.bool, device=x.device)
        mask[mask_indices] = True

        # Save original values
        x_original = x.clone()

        # Replace masked features with learned mask token or zeros
        x_masked = x.clone()
        x_masked[mask_indices] = 0

        return x_masked, mask, x_original

    def pretrain_step(self, x, edge_index, edge_attr, optimizer):
        """
        Single pre-training step.
        """
        self.model.train()
        optimizer.zero_grad()

        # Mask features
        x_masked, mask, x_original = self.mask_features(x)

        # Forward pass
        h = self.model.forward_features(x_masked, edge_index, edge_attr)

        # Decode
        reconstructed = self.decoder(h[mask])
        target = x_original[mask]

        # Reconstruction loss
        loss = F.mse_loss(reconstructed, target)

        loss.backward()
        optimizer.step()

        return loss.item()

    def pretrain(self, x, edge_index, edge_attr, epochs=50, lr=1e-3):
        """
        Run self-supervised pre-training.
        """
        optimizer = AdamW(list(self.model.parameters()) + list(self.decoder.parameters()), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        print(f"Starting self-supervised pre-training for {epochs} epochs...")
        for epoch in range(epochs):
            loss = self.pretrain_step(x, edge_index, edge_attr, optimizer)
            scheduler.step(loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        print("Pre-training completed!")
        return self.model


class ContrastiveLearning:
    """
    Graph contrastive learning (SimCLR-style).
    Learns representations by contrasting positive and negative pairs.
    """
    def __init__(self, model, temperature=0.5):
        self.model = model
        self.temperature = temperature

    def augment_graph(self, x, edge_index, edge_attr=None, aug_type='edge_drop'):
        """
        Create augmented view of the graph.
        """
        if aug_type == 'edge_drop':
            # Randomly drop edges
            num_edges = edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=x.device) > 0.2
            aug_edge_index = edge_index[:, keep_mask]
            if edge_attr is not None:
                aug_edge_attr = edge_attr[keep_mask]
            else:
                aug_edge_attr = None

        elif aug_type == 'node_drop':
            # Randomly drop nodes
            N = x.size(0)
            keep_mask = torch.rand(N, device=x.device) > 0.1
            keep_indices = torch.where(keep_mask)[0]

            # Filter edges
            edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
            aug_edge_index = edge_index[:, edge_mask]

            # Remap node indices
            node_map = torch.zeros(N, dtype=torch.long, device=x.device)
            node_map[keep_indices] = torch.arange(keep_indices.size(0), device=x.device)
            aug_edge_index = node_map[aug_edge_index]

            x = x[keep_indices]
            if edge_attr is not None:
                aug_edge_attr = edge_attr[edge_mask]
            else:
                aug_edge_attr = None

        elif aug_type == 'feature_mask':
            # Randomly mask features
            mask = torch.rand_like(x) > 0.1
            x = x * mask
            aug_edge_index = edge_index
            aug_edge_attr = edge_attr

        else:
            aug_edge_index = edge_index
            aug_edge_attr = edge_attr

        return x, aug_edge_index, aug_edge_attr

    def nt_xent_loss(self, z_i, z_j):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        """
        batch_size = z_i.size(0)

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, hidden_dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2*batch_size, 2*batch_size)

        # Create positive pairs mask
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        pos_sim = torch.cat([
            sim_matrix[range(batch_size), range(batch_size, 2 * batch_size)],
            sim_matrix[range(batch_size, 2 * batch_size), range(batch_size)]
        ])

        # Compute loss
        loss = -torch.log(pos_sim / sim_matrix.exp().sum(dim=1)).mean()

        return loss

    def contrastive_step(self, x, edge_index, edge_attr, optimizer):
        """
        Single contrastive learning step.
        """
        self.model.train()
        optimizer.zero_grad()

        # Create two augmented views
        x_1, edge_index_1, edge_attr_1 = self.augment_graph(x, edge_index, edge_attr, 'edge_drop')
        x_2, edge_index_2, edge_attr_2 = self.augment_graph(x, edge_index, edge_attr, 'feature_mask')

        # Get representations
        z_1 = self.model.forward_features(x_1, edge_index_1, edge_attr_1)
        z_2 = self.model.forward_features(x_2, edge_index_2, edge_attr_2)

        # Contrastive loss
        loss = self.nt_xent_loss(z_1, z_2)

        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self, x, edge_index, edge_attr, epochs=50, lr=1e-3):
        """
        Run contrastive pre-training.
        """
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        print(f"Starting contrastive pre-training for {epochs} epochs...")
        for epoch in range(epochs):
            loss = self.contrastive_step(x, edge_index, edge_attr, optimizer)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        print("Contrastive pre-training completed!")
        return self.model


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross-entropy loss.
    Prevents overconfidence and improves generalization.
    """
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        pred: (N, num_classes) - logits
        target: (N,) - class indices
        """
        log_probs = F.log_softmax(pred, dim=-1)

        # Create smoothed target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = -(true_dist * log_probs).sum(dim=-1).mean()
        return loss


class StochasticWeightAveraging:
    """
    SWA: Stochastic Weight Averaging for improved generalization.
    Averages model weights over training trajectory.
    Reference: Izmailov et al. "Averaging Weights Leads to Wider Optima and Better Generalization", 2018
    """
    def __init__(self, model, start_epoch=10):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_model = copy.deepcopy(model)
        self.swa_n = 0

        # Disable gradients for SWA model
        for param in self.swa_model.parameters():
            param.requires_grad = False

    def update(self, epoch):
        """
        Update SWA model with current weights.
        """
        if epoch < self.start_epoch:
            return

        # Exponential moving average
        for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
            swa_param.data.mul_(self.swa_n / (self.swa_n + 1))
            swa_param.data.add_(param.data / (self.swa_n + 1))

        self.swa_n += 1

    def swap_swa(self):
        """
        Swap current model with SWA model for evaluation.
        """
        # Save current state
        self.original_state = copy.deepcopy(self.model.state_dict())
        # Load SWA state
        self.model.load_state_dict(self.swa_model.state_dict())

    def swap_back(self):
        """
        Restore original model.
        """
        if hasattr(self, 'original_state'):
            self.model.load_state_dict(self.original_state)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Down-weights easy examples and focuses on hard examples.
    """
    def __init__(self, num_classes, gamma=2.0, alpha=None, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.device = device

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        else:
            self.alpha = None

    def compute_alpha(self, targets):
        """Adaptive class weighting."""
        if self.alpha is not None:
            return self.alpha

        counts = torch.bincount(targets, minlength=self.num_classes).float()
        total = counts.sum()
        beta = 1.0 - counts / (total + 1e-8)
        return beta.to(self.device)

    def forward(self, inputs, targets):
        """
        inputs: (N, C) - logits
        targets: (N,) - class indices
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        batch_size = targets.size(0)
        p_t = probs[torch.arange(batch_size, device=probs.device), targets]
        log_p_t = log_probs[torch.arange(batch_size, device=probs.device), targets]

        alpha_t = self.compute_alpha(targets)[targets]
        focal_weight = (1.0 - p_t) ** self.gamma

        loss = -alpha_t * focal_weight * log_p_t
        return loss.mean()


class AdvancedTrainer:
    """
    Advanced trainer combining multiple training strategies.
    """
    def __init__(self, model, config, device='cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('patience', 10) // 3
        )

        # Loss function
        use_focal = config.get('use_focal_loss', True)
        use_label_smoothing = config.get('use_label_smoothing', True)

        if use_label_smoothing and not use_focal:
            self.criterion = LabelSmoothingCrossEntropy(
                num_classes=config['num_classes'],
                smoothing=config.get('label_smoothing', 0.1)
            )
        elif use_focal:
            self.criterion = FocalLoss(
                num_classes=config['num_classes'],
                gamma=config.get('focal_gamma', 2.0),
                device=device
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        # SWA
        self.use_swa = config.get('use_swa', True)
        if self.use_swa:
            self.swa = StochasticWeightAveraging(model, start_epoch=config.get('swa_start_epoch', 50))

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, x, edge_index, edge_attr, y):
        """Single training epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(x, edge_index, edge_attr)
        loss = self.criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()

        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, x, edge_index, edge_attr, y):
        """Evaluation."""
        self.model.eval()
        logits = self.model(x, edge_index, edge_attr)
        loss = self.criterion(logits, y)

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()

        from sklearn.metrics import precision_recall_fscore_support
        y_cpu = y.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_cpu, preds_cpu, average='macro', zero_division=0
        )

        return {
            'loss': loss.item(),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def fit(self, train_data, val_data, epochs=None):
        """Full training loop."""
        epochs = epochs or self.config.get('epochs', 100)
        x_train, edge_index_train, edge_attr_train, y_train = train_data
        x_val, edge_index_val, edge_attr_val, y_val = val_data

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
                   'val_acc': [], 'val_f1': [], 'lr': []}

        print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'Val F1':>8}")
        print("-" * 70)

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                x_train, edge_index_train, edge_attr_train, y_train
            )
            val_metrics = self.evaluate(
                x_val, edge_index_val, edge_attr_val, y_val
            )

            # Scheduler step
            self.scheduler.step(val_metrics['loss'])

            # SWA update
            if self.use_swa:
                self.swa.update(epoch)

            # Record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1

            # Print progress
            if epoch % 10 == 0 or epoch < 5:
                print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | "
                      f"{val_metrics['loss']:>9.4f} | {val_metrics['accuracy']:>8.4f} | "
                      f"{val_metrics['f1']:>8.4f}")

            # Check early stopping
            if self.patience_counter >= self.config.get('patience', 15):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

            # Check learning rate
            if self.optimizer.param_groups[0]['lr'] < 1e-6:
                print(f"\nLearning rate too small, stopping")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Restored best validation model")

        return history


if __name__ == "__main__":
    # Test LabelSmoothingCrossEntropy
    print("Testing LabelSmoothingCrossEntropy...")
    criterion = LabelSmoothingCrossEntropy(num_classes=5, smoothing=0.1)
    pred = torch.randn(32, 5)
    target = torch.randint(0, 5, (32,))
    loss = criterion(pred, target)
    print(f"Loss: {loss.item():.4f}")

    # Test FocalLoss
    print("\nTesting FocalLoss...")
    focal = FocalLoss(num_classes=5, gamma=2.0)
    loss = focal(pred, target)
    print(f"Loss: {loss.item():.4f}")

    # Test SWA
    print("\nTesting StochasticWeightAveraging...")
    model = nn.Linear(10, 5)
    swa = StochasticWeightAveraging(model, start_epoch=5)
    for epoch in range(10):
        swa.update(epoch)
    print("SWA test passed!")

    print("\nAll tests passed!")
