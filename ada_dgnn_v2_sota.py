"""
Ada-DGNN V2: State-of-the-Art Network Intrusion Detection
==========================================================
Comprehensive optimization integrating 2024-2025 SOTA techniques:

Key Improvements over V1:
1. KAN (Kolmogorov-Arnold Networks): Replace standard MLPs with learnable activation functions
2. Multi-Scale GNN: Capture patterns at 1-hop, 2-hop, and 3-hop distances
3. Transformer Encoder: Global context modeling with self-attention
4. Borderline-SMOTE: Generate samples near decision boundary
5. ADASYN: Adaptive synthetic sampling based on difficulty
6. Advanced Training: SWA, Label Smoothing, Focal Loss
7. Feature Engineering: Statistical features and interactions

References:
- KAN: Liu et al., Nature 2025
- Multi-scale GNN: Various 2024 papers
- Borderline-SMOTE: Han et al., 2005
- ADASYN: He et al., 2008
- SWA: Izmailov et al., 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            classification_report, confusion_matrix, roc_auc_score,
                            f1_score, precision_score, recall_score)
from sklearn.neighbors import kneighbors_graph
from collections import Counter
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from kan_layer import FastKANLayer, KANLayer
from multiscale_gnn import EdgeAwareGATLayer, GraphTransformerEncoder, MultiScaleGNN
from data_augmentation import BorderlineSMOTE, ADASYN, FeatureEngineering, CICIDS2017Preprocessor
from training_strategies import AdvancedTrainer, FocalLoss, LabelSmoothingCrossEntropy

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Data
    'data_path': 'datasets/cicids2017.csv',
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,

    # Architecture
    'hidden_dim': 128,
    'num_gnn_layers': 2,
    'num_heads': 4,
    'dropout': 0.2,
    'edge_dim': 16,
    'use_kan': True,
    'use_transformer': True,
    'use_multiscale': True,
    'transformer_layers': 1,

    # Data Augmentation
    'use_smote': True,
    'smote_type': 'borderline',  # 'borderline' or 'adasyn'
    'smote_ratio': 0.5,
    'use_feature_engineering': True,

    # Training
    'epochs': 150,
    'batch_size': 256,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'patience': 20,
    'use_swa': True,
    'swa_start_epoch': 50,

    # Loss
    'use_focal_loss': True,
    'focal_gamma': 2.0,
    'use_label_smoothing': False,
    'label_smoothing': 0.1,

    # Graph
    'graph_k': 5,
    'graph_type': 'knn',

    # Device
    'device': 'auto',
}


# =============================================================================
# GRAPH BUILDING
# =============================================================================
class TrafficGraphBuilder:
    """Build graph structure from tabular features."""

    def __init__(self, k=5, edge_dim=16, device='cpu'):
        self.k = k
        self.edge_dim = edge_dim
        self.device = device

    def build_graph(self, X):
        """Build k-NN graph from features."""
        N = X.shape[0]

        # Build k-NN adjacency
        adj = kneighbors_graph(
            X, n_neighbors=min(self.k, N-1),
            mode='connectivity', metric='cosine', include_self=False
        )
        # Make symmetric by converting to array first
        adj = adj.toarray()
        adj = np.maximum(adj, adj.T)  # Symmetric

        # Extract edges
        rows, cols = adj.nonzero()
        edge_index = torch.LongTensor(np.array([rows, cols])).to(self.device)

        # Compute edge features
        X_t = torch.FloatTensor(X).to(self.device)
        xi = X_t[rows]
        xj = X_t[cols]

        # Cosine similarity
        norm_i = torch.norm(xi, dim=1, keepdim=True)
        norm_j = torch.norm(xj, dim=1, keepdim=True)
        cos_sim = (xi * xj).sum(dim=1) / (norm_i.squeeze() * norm_j.squeeze() + 1e-8)

        # Feature difference
        feat_diff = torch.abs(xi - xj)[:, :self.edge_dim-1]

        # Combine features
        if feat_diff.size(1) < self.edge_dim - 1:
            padding = torch.zeros(feat_diff.size(0), self.edge_dim - 1 - feat_diff.size(1),
                                device=self.device)
            feat_diff = torch.cat([feat_diff, padding], dim=1)

        edge_attr = torch.cat([cos_sim.unsqueeze(1), feat_diff], dim=1)

        return edge_index, edge_attr, torch.FloatTensor(X).to(self.device)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class AdaDGNN_V2(nn.Module):
    """
    Ada-DGNN V2: Full SOTA architecture.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, config):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-scale GNN
        if config.get('use_multiscale', True):
            self.multiscale_gnn = MultiScaleGNN(
                hidden_dim, hidden_dim, hidden_dim,
                num_scales=3, num_heads=config['num_heads'],
                edge_dim=config['edge_dim'], dropout=config['dropout']
            )
        else:
            self.gnn_layers = nn.ModuleList()
            for i in range(config['num_gnn_layers']):
                self.gnn_layers.append(
                    EdgeAwareGATLayer(
                        hidden_dim, hidden_dim, config['edge_dim'],
                        num_heads=config['num_heads'],
                        dropout=config['dropout'], concat=False
                    )
                )

        # KAN layers
        if config.get('use_kan', True):
            kan_hidden = hidden_dim // 2
            self.kan_layers = nn.ModuleList([
                FastKANLayer(hidden_dim, kan_hidden, num_grids=8, dropout=config['dropout']),
                FastKANLayer(kan_hidden, hidden_dim, num_grids=8, dropout=config['dropout'])
            ])

        # Transformer encoder
        if config.get('use_transformer', True):
            self.transformer = GraphTransformerEncoder(
                hidden_dim, hidden_dim,
                num_layers=config.get('transformer_layers', 1),
                num_heads=config['num_heads'],
                dropout=config['dropout']
            )

        # Fusion layer
        fusion_dim = hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(config['dropout'])
        )

        # Classifier
        classifier_hidden = hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.SiLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(classifier_hidden, num_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None):
        # Input projection
        h = self.input_proj(x)
        h = F.silu(h)

        # Multi-scale GNN or standard GNN
        if self.config.get('use_multiscale', True):
            h = self.multiscale_gnn(h, edge_index, edge_attr)
        else:
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index, edge_attr)
                h = F.silu(h)

        # KAN layers
        if self.config.get('use_kan', True):
            for kan_layer in self.kan_layers:
                h = kan_layer(h)
                h = F.silu(h)

        # Transformer
        if self.config.get('use_transformer', True):
            h = self.transformer(h)

        # Fusion
        h = self.fusion(h)

        # Classification
        logits = self.classifier(h)
        return logits


# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def preprocess_data(config):
    """Load and preprocess data with augmentation."""
    data_path = config['data_path']

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}")
        print("Generating synthetic data for testing...")
        np.random.seed(42)
        N, D = 10000, 30
        X = np.random.randn(N, D)
        y = np.random.choice(5, N, p=[0.6, 0.2, 0.1, 0.06, 0.04])
        feature_cols = [f'feature_{i}' for i in range(D)]
    else:
        # Load real data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Unsupported file format")

        # Clean and preprocess
        preprocessor = CICIDS2017Preprocessor()
        X, y, feature_cols = preprocessor.preprocess(df)

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Class distribution: {Counter(y)}")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=config['test_size'] + config['val_size'],
        random_state=config['random_state'], stratify=y
    )
    val_ratio = config['val_size'] / (config['test_size'] + config['val_size'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio,
        random_state=config['random_state'], stratify=y_temp
    )

    # Data augmentation
    if config['use_smote']:
        print(f"Applying {config['smote_type']} SMOTE...")
        if config['smote_type'] == 'borderline':
            sampler = BorderlineSMOTE(random_state=config['random_state'])
        else:
            sampler = ADASYN(random_state=config['random_state'])

        X_train, y_train = sampler.fit_resample(
            X_train, y_train, target_ratio=config['smote_ratio']
        )
        print(f"After augmentation: {Counter(y_train)}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'num_classes': len(np.unique(y)),
        'feature_cols': feature_cols
    }


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================
def train_model(model, train_data, val_data, config, device):
    """Train the model."""
    x_train, edge_index_train, edge_attr_train, y_train = train_data
    x_val, edge_index_val, edge_attr_val, y_val = val_data

    # Setup trainer
    trainer_config = {
        'num_classes': config['num_classes'],
        'learning_rate': config['learning_rate'],
        'weight_decay': config['weight_decay'],
        'use_focal_loss': config['use_focal_loss'],
        'focal_gamma': config['focal_gamma'],
        'use_label_smoothing': config['use_label_smoothing'],
        'label_smoothing': config['label_smoothing'],
        'patience': config['patience'],
        'use_swa': config['use_swa'],
        'swa_start_epoch': config['swa_start_epoch'],
        'epochs': config['epochs']
    }

    trainer = AdvancedTrainer(model, trainer_config, device)

    print("\n" + "="*70)
    print("Training ADA-DGNN V2")
    print("="*70)

    history = trainer.fit(train_data, val_data, epochs=config['epochs'])

    return trainer, history


def evaluate_model(model, x, edge_index, edge_attr, y_true, class_names=None):
    """Comprehensive model evaluation."""
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        preds = logits.argmax(dim=-1).cpu().numpy()
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    y_true = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true

    # Basic metrics
    acc = accuracy_score(y_true, preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, average=None, zero_division=0
    )
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    weighted_f1 = f1_score(y_true, preds, average='weighted')

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")

    print("\nPer-Class Metrics:")
    print("-"*70)
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        name = class_names[i] if class_names else f"Class {i}"
        print(f"{name:20s} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f} | Support: {int(s)}")

    # Classification report
    print("\nClassification Report:")
    target_names = class_names if class_names else None
    print(classification_report(y_true, preds, target_names=target_names, digits=4))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, preds)
    print(cm)

    # AUROC (if applicable)
    if len(np.unique(y_true)) > 1 and probs.shape[1] > 1:
        try:
            if probs.shape[1] == 2:
                auroc = roc_auc_score(y_true, probs[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_true)
                y_bin = label_binarize(y_true, classes=classes)
                auroc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
            print(f"\nMacro AUROC: {auroc:.4f}")
        except Exception as e:
            auroc = None
            print(f"Could not compute AUROC: {e}")

    print("="*70)

    return {
        'accuracy': acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'predictions': preds.tolist(),
        'probabilities': probs.tolist()
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("="*70)
    print("ADA-DGNN V2: State-of-the-Art Network Intrusion Detection")
    print("="*70)

    # Device setup
    if CONFIG['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(CONFIG['device'])

    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Preprocess data
    print("\n" + "-"*70)
    print("Loading and preprocessing data...")
    data = preprocess_data(CONFIG)
    CONFIG['num_classes'] = data['num_classes']

    # Build graphs
    print("\nBuilding graphs...")
    graph_builder = TrafficGraphBuilder(k=CONFIG['graph_k'], edge_dim=CONFIG['edge_dim'], device=device)

    edge_index_train, edge_attr_train, x_train = graph_builder.build_graph(data['X_train'])
    edge_index_val, edge_attr_val, x_val = graph_builder.build_graph(data['X_val'])
    edge_index_test, edge_attr_test, x_test = graph_builder.build_graph(data['X_test'])

    y_train = torch.LongTensor(data['y_train']).to(device)
    y_val = torch.LongTensor(data['y_val']).to(device)
    y_test = torch.LongTensor(data['y_test']).to(device)

    print(f"Train: {x_train.size(0)} nodes, {edge_index_train.size(1)} edges")
    print(f"Val: {x_val.size(0)} nodes, {edge_index_val.size(1)} edges")
    print(f"Test: {x_test.size(0)} nodes, {edge_index_test.size(1)} edges")

    # Create model
    print("\n" + "-"*70)
    print("Creating model...")
    model = AdaDGNN_V2(
        input_dim=x_train.size(1),
        hidden_dim=CONFIG['hidden_dim'],
        num_classes=CONFIG['num_classes'],
        config=CONFIG
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    train_data = (x_train, edge_index_train, edge_attr_train, y_train)
    val_data = (x_val, edge_index_val, edge_attr_val, y_val)

    trainer, history = train_model(model, train_data, val_data, CONFIG, device)

    # Evaluate on test set
    print("\n" + "-"*70)
    print("Final Test Set Evaluation")
    test_metrics = evaluate_model(
        model, x_test, edge_index_test, edge_attr_test, y_test,
        class_names=[f"Class_{i}" for i in range(CONFIG['num_classes'])]
    )

    # Save model
    save_path = 'ada_dgnn_v2_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'history': history,
        'test_metrics': test_metrics
    }, save_path)
    print(f"\nModel saved to {save_path}")

    # Save results
    results = {
        'config': CONFIG,
        'test_metrics': test_metrics,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }

    with open('ada_dgnn_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to ada_dgnn_v2_results.json")
    print("\n" + "="*70)
    print("ADA-DGNN V2 Training Complete!")
    print("="*70)

    return model, history, test_metrics


if __name__ == "__main__":
    main()
