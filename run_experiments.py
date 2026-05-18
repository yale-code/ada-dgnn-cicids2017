"""
Comprehensive Experiments for ADA-DGNN V2
==========================================
Runs ablation studies and comparison experiments.

Experiments:
1. Baseline (original model without enhancements)
2. KAN only
3. Multi-scale GNN only
4. Transformer only
5. Borderline-SMOTE only
6. SWA only
7. Full model (all enhancements)
8. Optimal configuration (from hyperparameter search)

Outputs:
- Comparison table
- Learning curves
- Ablation study results
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from ada_dgnn_v2_sota import AdaDGNN_V2, preprocess_data, evaluate_model, TrafficGraphBuilder, CONFIG as BASE_CONFIG

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ExperimentRunner:
    """Run and compare different model configurations."""

    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device
        self.results = []

        # Prepare data
        self.graph_builder = TrafficGraphBuilder(k=5, edge_dim=16, device=device)

        self.edge_index_train, self.edge_attr_train, self.x_train = \
            self.graph_builder.build_graph(data['X_train'])
        self.edge_index_val, self.edge_attr_val, self.x_val = \
            self.graph_builder.build_graph(data['X_val'])
        self.edge_index_test, self.edge_attr_test, self.x_test = \
            self.graph_builder.build_graph(data['X_test'])

        self.y_train = torch.LongTensor(data['y_train']).to(device)
        self.y_val = torch.LongTensor(data['y_val']).to(device)
        self.y_test = torch.LongTensor(data['y_test']).to(device)

    def run_experiment(self, name, config):
        """Run a single experiment."""
        print(f"\n{'='*70}")
        print(f"Experiment: {name}")
        print(f"{'='*70}")

        # Create model
        model = AdaDGNN_V2(
            input_dim=self.x_train.size(1),
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            config=config
        ).to(self.device)

        # Train
        from training_strategies import AdvancedTrainer

        trainer_config = {
            'num_classes': config['num_classes'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'use_focal_loss': config.get('use_focal_loss', True),
            'focal_gamma': config.get('focal_gamma', 2.0),
            'patience': config.get('patience', 15),
            'use_swa': config.get('use_swa', False),
            'epochs': config.get('epochs', 50)
        }

        trainer = AdvancedTrainer(model, trainer_config, self.device)

        train_data = (self.x_train, self.edge_index_train, self.edge_attr_train, self.y_train)
        val_data = (self.x_val, self.edge_index_val, self.edge_attr_val, self.y_val)

        start_time = time.time()
        history = trainer.fit(train_data, val_data, epochs=config.get('epochs', 50))
        train_time = time.time() - start_time

        # Evaluate
        test_metrics = evaluate_model(
            model, self.x_test, self.edge_index_test, self.edge_attr_test, self.y_test
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Record results
        result = {
            'name': name,
            'accuracy': test_metrics['accuracy'],
            'macro_f1': test_metrics['macro_f1'],
            'macro_precision': test_metrics['macro_precision'],
            'macro_recall': test_metrics['macro_recall'],
            'weighted_f1': test_metrics['weighted_f1'],
            'train_time': train_time,
            'total_params': total_params,
            'config': config,
            'history': history
        }

        self.results.append(result)

        print(f"\nResults for {name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Macro F1: {result['macro_f1']:.4f}")
        print(f"  Train Time: {result['train_time']:.1f}s")

        return result

    def run_all_experiments(self):
        """Run all ablation experiments."""
        print("\n" + "="*70)
        print("RUNNING ALL EXPERIMENTS")
        print("="*70)

        # 1. Baseline (minimal configuration)
        baseline_config = BASE_CONFIG.copy()
        baseline_config.update({
            'use_kan': False,
            'use_transformer': False,
            'use_multiscale': False,
            'use_smote': False,
            'use_swa': False,
            'epochs': 50
        })
        self.run_experiment('Baseline', baseline_config)

        # 2. KAN only
        kan_config = BASE_CONFIG.copy()
        kan_config.update({
            'use_kan': True,
            'use_transformer': False,
            'use_multiscale': False,
            'use_smote': False,
            'use_swa': False,
            'epochs': 50
        })
        self.run_experiment('KAN Only', kan_config)

        # 3. Multi-scale GNN only
        multiscale_config = BASE_CONFIG.copy()
        multiscale_config.update({
            'use_kan': False,
            'use_transformer': False,
            'use_multiscale': True,
            'use_smote': False,
            'use_swa': False,
            'epochs': 50
        })
        self.run_experiment('Multi-scale GNN Only', multiscale_config)

        # 4. Transformer only
        transformer_config = BASE_CONFIG.copy()
        transformer_config.update({
            'use_kan': False,
            'use_transformer': True,
            'use_multiscale': False,
            'use_smote': False,
            'use_swa': False,
            'epochs': 50
        })
        self.run_experiment('Transformer Only', transformer_config)

        # 5. Borderline-SMOTE only
        smote_config = BASE_CONFIG.copy()
        smote_config.update({
            'use_kan': False,
            'use_transformer': False,
            'use_multiscale': False,
            'use_smote': True,
            'smote_type': 'borderline',
            'use_swa': False,
            'epochs': 50
        })
        self.run_experiment('Borderline-SMOTE Only', smote_config)

        # 6. SWA only
        swa_config = BASE_CONFIG.copy()
        swa_config.update({
            'use_kan': False,
            'use_transformer': False,
            'use_multiscale': False,
            'use_smote': False,
            'use_swa': True,
            'swa_start_epoch': 30,
            'epochs': 50
        })
        self.run_experiment('SWA Only', swa_config)

        # 7. Full model (all enhancements)
        full_config = BASE_CONFIG.copy()
        full_config.update({
            'use_kan': True,
            'use_transformer': True,
            'use_multiscale': True,
            'use_smote': True,
            'smote_type': 'borderline',
            'use_swa': True,
            'epochs': 50
        })
        self.run_experiment('Full Model (All)', full_config)

        # 8. Without KAN (ablation)
        no_kan_config = BASE_CONFIG.copy()
        no_kan_config.update({
            'use_kan': False,
            'use_transformer': True,
            'use_multiscale': True,
            'use_smote': True,
            'use_swa': True,
            'epochs': 50
        })
        self.run_experiment('All except KAN', no_kan_config)

        # 9. Without Transformer
        no_transformer_config = BASE_CONFIG.copy()
        no_transformer_config.update({
            'use_kan': True,
            'use_transformer': False,
            'use_multiscale': True,
            'use_smote': True,
            'use_swa': True,
            'epochs': 50
        })
        self.run_experiment('All except Transformer', no_transformer_config)

        # 10. Without Multi-scale
        no_multiscale_config = BASE_CONFIG.copy()
        no_multiscale_config.update({
            'use_kan': True,
            'use_transformer': True,
            'use_multiscale': False,
            'use_smote': True,
            'use_swa': True,
            'epochs': 50
        })
        self.run_experiment('All except Multi-scale', no_multiscale_config)

    def generate_comparison_table(self):
        """Generate comparison table."""
        print("\n" + "="*70)
        print("COMPARISON TABLE")
        print("="*70)

        df = pd.DataFrame([
            {
                'Model': r['name'],
                'Accuracy': f"{r['accuracy']:.4f}",
                'Macro F1': f"{r['macro_f1']:.4f}",
                'Precision': f"{r['macro_precision']:.4f}",
                'Recall': f"{r['macro_recall']:.4f}",
                'Train Time (s)': f"{r['train_time']:.1f}",
                'Parameters': f"{r['total_params']:,}"
            }
            for r in self.results
        ])

        print(df.to_string(index=False))

        # Save to CSV
        df.to_csv('experiments/comparison_table.csv', index=False)
        print("\nSaved to experiments/comparison_table.csv")

        return df

    def plot_learning_curves(self):
        """Plot learning curves for all experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for result in self.results:
            history = result['history']
            name = result['name']

            # Train loss
            axes[0, 0].plot(history['train_loss'], label=name, alpha=0.7)
            # Val loss
            axes[0, 1].plot(history['val_loss'], label=name, alpha=0.7)
            # Train accuracy
            axes[1, 0].plot(history['train_acc'], label=name, alpha=0.7)
            # Val F1
            axes[1, 1].plot(history['val_f1'], label=name, alpha=0.7)

        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('experiments/learning_curves.png', dpi=300, bbox_inches='tight')
        print("\nLearning curves saved to experiments/learning_curves.png")
        plt.close()

    def plot_ablation_study(self):
        """Plot ablation study results."""
        # Filter ablation experiments
        ablation_names = ['Full Model (All)', 'All except KAN',
                         'All except Transformer', 'All except Multi-scale']
        ablation_results = [r for r in self.results if r['name'] in ablation_names]

        if len(ablation_results) < 2:
            print("Not enough ablation experiments to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        names = [r['name'] for r in ablation_results]
        f1_scores = [r['macro_f1'] for r in ablation_results]

        colors = ['#2ecc71' if 'Full' in n else '#e74c3c' for n in names]

        bars = ax.barh(names, f1_scores, color=colors, alpha=0.7)
        ax.set_xlabel('Macro F1 Score', fontsize=12)
        ax.set_title('Ablation Study: Impact of Individual Components', fontsize=14, fontweight='bold')
        ax.set_xlim([min(f1_scores) * 0.95, min(1.0, max(f1_scores) * 1.05)])

        # Add value labels
        for bar, score in zip(bars, f1_scores):
            ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{score:.4f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('experiments/ablation_study.png', dpi=300, bbox_inches='tight')
        print("Ablation study plot saved to experiments/ablation_study.png")
        plt.close()

    def plot_comparison_bars(self):
        """Plot bar chart comparing all models."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        names = [r['name'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        f1_scores = [r['macro_f1'] for r in self.results]

        x = np.arange(len(names))
        width = 0.35

        # Accuracy
        axes[0].bar(x, accuracies, color='#3498db', alpha=0.7)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Comparison: Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[0].grid(True, alpha=0.3, axis='y')

        # F1 Score
        axes[1].bar(x, f1_scores, color='#e74c3c', alpha=0.7)
        axes[1].set_ylabel('Macro F1 Score', fontsize=12)
        axes[1].set_title('Model Comparison: F1 Score', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('experiments/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison plot saved to experiments/model_comparison.png")
        plt.close()

    def save_results(self):
        """Save all results to JSON."""
        results_data = {
            'experiments': [
                {
                    'name': r['name'],
                    'accuracy': r['accuracy'],
                    'macro_f1': r['macro_f1'],
                    'macro_precision': r['macro_precision'],
                    'macro_recall': r['macro_recall'],
                    'weighted_f1': r['weighted_f1'],
                    'train_time': r['train_time'],
                    'total_params': r['total_params']
                }
                for r in self.results
            ]
        }

        with open('experiments/experiment_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)

        print("\nResults saved to experiments/experiment_results.json")


def main():
    print("="*70)
    print("ADA-DGNN V2: Comprehensive Experiments")
    print("="*70)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Preprocess data
    print("\nLoading data...")
    data = preprocess_data(BASE_CONFIG)

    # Create output directory
    os.makedirs('experiments', exist_ok=True)

    # Run experiments
    runner = ExperimentRunner(data, device)
    runner.run_all_experiments()

    # Generate outputs
    runner.generate_comparison_table()
    runner.plot_learning_curves()
    runner.plot_ablation_study()
    runner.plot_comparison_bars()
    runner.save_results()

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nResults saved in experiments/ directory:")
    print("  - comparison_table.csv")
    print("  - experiment_results.json")
    print("  - learning_curves.png")
    print("  - ablation_study.png")
    print("  - model_comparison.png")


if __name__ == "__main__":
    main()
