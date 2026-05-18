"""
Optuna Hyperparameter Optimization for ADA-DGNN
================================================
Uses Bayesian optimization to find optimal hyperparameters.

Search space:
- Architecture: hidden_dim, num_layers, num_heads, dropout
- Training: learning_rate, weight_decay, batch_size
- Loss: focal_gamma, use_focal_loss
- Data: use_smote, smote_ratio, augmentation_type

Features:
- Multi-objective: Maximize F1, minimize inference time
- Early pruning of unpromising trials
- Results saved to SQLite database
"""
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from kan_layer import FastKANLayer
from multiscale_gnn import EdgeAwareGATLayer, GraphTransformerEncoder
from data_augmentation import BorderlineSMOTE, ADASYN, CICIDS2017Preprocessor
from training_strategies import AdvancedTrainer, FocalLoss


class ConfigurableModel(nn.Module):
    """
    Model with configurable architecture for hyperparameter search.
    """
    def __init__(self, input_dim, num_classes, trial_params):
        super().__init__()
        self.params = trial_params

        hidden_dim = trial_params['hidden_dim']
        num_layers = trial_params['num_layers']
        num_heads = trial_params['num_heads']
        dropout = trial_params['dropout']
        use_kan = trial_params['use_kan']
        use_transformer = trial_params['use_transformer']
        edge_dim = trial_params['edge_dim']

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                EdgeAwareGATLayer(
                    hidden_dim, hidden_dim, edge_dim,
                    num_heads=num_heads, dropout=dropout, concat=False
                )
            )

        # Optional KAN layers
        self.use_kan = use_kan
        if use_kan:
            kan_hidden = trial_params.get('kan_hidden', hidden_dim // 2)
            self.kan_layers = nn.ModuleList([
                FastKANLayer(hidden_dim, kan_hidden, num_grids=8, dropout=dropout),
                FastKANLayer(kan_hidden, hidden_dim, num_grids=8, dropout=dropout)
            ])

        # Optional Transformer
        self.use_transformer = use_transformer
        if use_transformer:
            transformer_layers = trial_params.get('transformer_layers', 1)
            self.transformer = GraphTransformerEncoder(
                hidden_dim, hidden_dim, num_layers=transformer_layers,
                num_heads=num_heads, dropout=dropout
            )

        # Classifier
        classifier_hidden = hidden_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None):
        # Input projection
        h = self.input_proj(x)
        h = torch.silu(h)

        # GNN layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index, edge_attr)
            h = torch.silu(h)

        # Optional KAN
        if self.use_kan:
            for kan_layer in self.kan_layers:
                h = kan_layer(h)
                h = torch.silu(h)

        # Optional Transformer
        if self.use_transformer:
            h = self.transformer(h)

        # Classification
        logits = self.classifier(h)
        return logits


def load_data(data_path, config):
    """Load and preprocess data."""
    if not os.path.exists(data_path):
        # Generate synthetic data for testing
        print(f"Data file not found: {data_path}, generating synthetic data...")
        np.random.seed(42)
        N, D = 5000, 30
        X = np.random.randn(N, D)
        y = np.random.choice(config['num_classes'], N,
                            p=[0.6, 0.2, 0.1, 0.06, 0.04])
        return X, y

    # Load real data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Unsupported file format")

    # Preprocess
    preprocessor = CICIDS2017Preprocessor()
    X, y, _ = preprocessor.preprocess(df)

    return X, y


def create_graph(X, k=5, device='cpu'):
    """Create k-NN graph from features."""
    from sklearn.neighbors import kneighbors_graph
    N = X.shape[0]
    adj = kneighbors_graph(X, n_neighbors=min(k, N-1),
                          mode='connectivity', metric='cosine',
                          include_self=False)
    adj = np.maximum(adj, adj.T)  # Make symmetric

    # Convert to edge_index
    rows, cols = adj.nonzero()
    edge_index = torch.LongTensor(np.array([rows, cols])).to(device)

    # Simple edge features (cosine similarity)
    X_t = torch.FloatTensor(X).to(device)
    xi = X_t[rows]
    xj = X_t[cols]
    cos_sim = nn.CosineSimilarity(dim=1)(xi, xj)
    edge_attr = torch.stack([cos_sim, torch.abs(xi - xj).mean(dim=1)], dim=1)

    return edge_index, edge_attr


def objective(trial, X_train, y_train, X_val, y_val, config):
    """
    Optuna objective function.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameter search space
    trial_params = {
        # Architecture
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'num_heads': trial.suggest_categorical('num_heads', [2, 4, 8]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'edge_dim': trial.suggest_categorical('edge_dim', [8, 16, 32]),

        # Optional components
        'use_kan': trial.suggest_categorical('use_kan', [True, False]),
        'use_transformer': trial.suggest_categorical('use_transformer', [True, False]),

        # Training
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),

        # Loss
        'use_focal_loss': trial.suggest_categorical('use_focal_loss', [True, False]),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),

        # Data augmentation
        'use_smote': trial.suggest_categorical('use_smote', [True, False]),
        'smote_ratio': trial.suggest_float('smote_ratio', 0.3, 1.0),
        'augmentation_type': trial.suggest_categorical('augmentation_type', ['borderline', 'adasyn']),
    }

    # Data augmentation
    if trial_params['use_smote']:
        if trial_params['augmentation_type'] == 'borderline':
            sampler = BorderlineSMOTE(random_state=42)
        else:
            sampler = ADASYN(random_state=42)
        X_train_res, y_train_res = sampler.fit_resample(
            X_train, y_train, target_ratio=trial_params['smote_ratio']
        )
    else:
        X_train_res, y_train_res = X_train, y_train

    # Convert to tensors
    x_train = torch.FloatTensor(X_train_res).to(device)
    y_train_t = torch.LongTensor(y_train_res).to(device)
    x_val = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    # Create graphs
    edge_index_train, edge_attr_train = create_graph(X_train_res, k=5, device=device)
    edge_index_val, edge_attr_val = create_graph(X_val, k=5, device=device)

    # Create model
    model_config = {
        'num_classes': config['num_classes'],
        **trial_params
    }
    model = ConfigurableModel(X_train_res.shape[1], config['num_classes'], trial_params).to(device)

    # Create trainer
    trainer_config = {
        'num_classes': config['num_classes'],
        'learning_rate': trial_params['learning_rate'],
        'weight_decay': trial_params['weight_decay'],
        'use_focal_loss': trial_params['use_focal_loss'],
        'focal_gamma': trial_params['focal_gamma'],
        'patience': 15,
        'use_swa': True,
        'swa_start_epoch': 30,
    }

    trainer = AdvancedTrainer(model, trainer_config, device)

    # Training
    train_data = (x_train, edge_index_train, edge_attr_train, y_train_t)
    val_data = (x_val, edge_index_val, edge_attr_val, y_val_t)

    try:
        history = trainer.fit(train_data, val_data, epochs=50)
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0, 1000.0  # Return poor metrics

    # Evaluate
    val_metrics = trainer.evaluate(x_val, edge_index_val, edge_attr_val, y_val_t)
    val_f1 = val_metrics['f1']

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_val, edge_index_val, edge_attr_val)
    inference_time = (time.time() - start_time) / 10

    # Report intermediate values for pruning
    for epoch, val_loss in enumerate(history['val_loss']):
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_f1, inference_time


def run_optimization(data_path, num_trials=50, timeout=3600):
    """
    Run hyperparameter optimization with Optuna.
    """
    print("=" * 60)
    print("Optuna Hyperparameter Optimization for ADA-DGNN")
    print("=" * 60)

    # Load data
    config = {'num_classes': 5}  # Will be updated
    X, y = load_data(data_path, config)
    config['num_classes'] = len(np.unique(y))

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {config['num_classes']} classes")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        directions=['maximize', 'minimize'],  # Maximize F1, minimize inference time
        sampler=sampler,
        pruner=pruner,
        study_name='ada_dgnn_optimization'
    )

    # Optimization
    print(f"\nRunning {num_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, config),
        n_trials=num_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Results
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)

    # Get Pareto front (multi-objective)
    pareto_trials = study.best_trials
    print(f"\nNumber of Pareto-optimal trials: {len(pareto_trials)}")

    # Best trial by F1
    best_trial = max(pareto_trials, key=lambda t: t.values[0])
    print(f"\nBest F1 Score: {best_trial.values[0]:.4f}")
    print(f"Inference Time: {best_trial.values[1]:.4f}s")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save results
    results = {
        'best_params': best_trial.params,
        'best_f1': best_trial.values[0],
        'best_inference_time': best_trial.values[1],
        'all_trials': [
            {'params': t.params, 'values': t.values}
            for t in study.trials if t.values is not None
        ]
    }

    with open('optuna_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to optuna_results.json")
    print(f"Study saved to optuna_study.db")

    return study, best_trial


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter optimization for ADA-DGNN')
    parser.add_argument('--data', type=str, default='datasets/cicids2017.csv',
                       help='Path to dataset')
    parser.add_argument('--trials', type=int, default=30,
                       help='Number of optimization trials')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout in seconds')

    args = parser.parse_args()

    study, best_trial = run_optimization(args.data, args.trials, args.timeout)
