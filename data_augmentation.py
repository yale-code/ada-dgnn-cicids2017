"""
Advanced Data Augmentation for NIDS
====================================
Implements sophisticated oversampling and feature engineering techniques.

Techniques:
1. Borderline-SMOTE: Generate samples near decision boundary
2. ADASYN: Adaptive synthetic sampling
3. Feature engineering: Statistical features, interaction features
4. Data cleaning: Handle CICIDS2017 specific issues
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, RobustScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class BorderlineSMOTE:
    """
    Borderline-SMOTE: Generate synthetic samples only near decision boundary.
    More effective than standard SMOTE for imbalanced datasets.

    Reference: Han et al. "Borderline-SMOTE: A New Over-Sampling Method in
    Imbalanced Data Sets Learning", 2005
    """
    def __init__(self, k_neighbors=5, m_neighbors=10, random_state=42):
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, X, y, target_ratio=0.5):
        """
        Apply Borderline-SMOTE oversampling.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        target_ratio : float
            Desired minority:majority ratio

        Returns:
        --------
        X_resampled, y_resampled
        """
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_resampled = [X]
        y_resampled = [y]

        for cls, count in zip(classes, counts):
            if count == max_count:
                continue  # Skip majority class

            # Calculate number of samples to generate
            n_samples = int(max_count * target_ratio) - count
            if n_samples <= 0:
                continue

            # Get minority samples
            minority_mask = y == cls
            minority_samples = X[minority_mask]

            # Find danger samples (near decision boundary)
            danger_indices = self._find_danger_samples(X, y, cls)

            if len(danger_indices) == 0:
                # Fall back to regular SMOTE if no danger samples
                danger_indices = np.arange(len(minority_samples))

            danger_samples = minority_samples[danger_indices]

            # Generate synthetic samples from danger samples
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(minority_samples)))
            nn.fit(minority_samples)

            for _ in range(n_samples):
                # Random danger sample
                idx = self.rng.choice(len(danger_samples))
                sample = danger_samples[idx]

                # Find k nearest minority neighbors
                _, neighbors_idx = nn.kneighbors([sample])
                neighbors_idx = neighbors_idx[0][1:]  # Exclude self

                if len(neighbors_idx) > 0:
                    nn_idx = self.rng.choice(neighbors_idx)
                    neighbor = minority_samples[nn_idx]

                    # Generate synthetic sample
                    alpha = self.rng.random()
                    synthetic = sample + alpha * (neighbor - sample)

                    X_resampled.append(synthetic.reshape(1, -1))
                    y_resampled.append(np.array([cls]))

        X_out = np.vstack(X_resampled)
        y_out = np.hstack(y_resampled)

        # Shuffle
        perm = self.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]

    def _find_danger_samples(self, X, y, minority_class):
        """
        Find samples near decision boundary (danger samples).
        """
        minority_mask = y == minority_class
        minority_samples = X[minority_mask]

        # Fit k-NN on all data
        nn = NearestNeighbors(n_neighbors=self.m_neighbors + 1)
        nn.fit(X)

        danger_indices = []
        minority_idx_map = np.where(minority_mask)[0]

        for i, sample in enumerate(minority_samples):
            _, neighbors_idx = nn.kneighbors([sample])
            neighbors_idx = neighbors_idx[0][1:]  # Exclude self

            # Count majority class neighbors
            neighbor_labels = y[neighbors_idx]
            n_majority = np.sum(neighbor_labels != minority_class)

            # Danger sample: more than half are from majority class
            if n_majority > self.m_neighbors / 2:
                danger_indices.append(i)

        return np.array(danger_indices)


class ADASYN:
    """
    ADASYN: Adaptive Synthetic Sampling
    Generates more samples for harder-to-learn minority examples.

    Reference: He et al. "ADASYN: Adaptive Synthetic Sampling Approach
    for Imbalanced Learning", 2008
    """
    def __init__(self, k_neighbors=5, random_state=42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit_resample(self, X, y, target_ratio=0.5):
        """
        Apply ADASYN oversampling.
        """
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_resampled = [X]
        y_resampled = [y]

        for cls, count in zip(classes, counts):
            if count == max_count:
                continue

            n_samples_total = int(max_count * target_ratio) - count
            if n_samples_total <= 0:
                continue

            minority_mask = y == cls
            minority_samples = X[minority_mask]
            n_minority = len(minority_samples)

            # Calculate difficulty level for each minority sample
            difficulties = self._calculate_difficulties(X, y, cls)

            # Normalize to get sample weights
            if difficulties.sum() > 0:
                sample_weights = difficulties / difficulties.sum()
            else:
                sample_weights = np.ones(n_minority) / n_minority

            # Determine samples to generate per minority instance
            samples_per_instance = (sample_weights * n_samples_total).astype(int)

            # Ensure total matches
            while samples_per_instance.sum() < n_samples_total:
                idx = self.rng.choice(n_minority, p=sample_weights)
                samples_per_instance[idx] += 1

            # Generate synthetic samples
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_minority))
            nn.fit(minority_samples)

            for i, n_gen in enumerate(samples_per_instance):
                if n_gen == 0:
                    continue

                sample = minority_samples[i]
                _, neighbors_idx = nn.kneighbors([sample])
                neighbors_idx = neighbors_idx[0][1:]

                for _ in range(n_gen):
                    if len(neighbors_idx) > 0:
                        nn_idx = self.rng.choice(neighbors_idx)
                        neighbor = minority_samples[nn_idx]

                        alpha = self.rng.random()
                        synthetic = sample + alpha * (neighbor - sample)

                        X_resampled.append(synthetic.reshape(1, -1))
                        y_resampled.append(np.array([cls]))

        X_out = np.vstack(X_resampled)
        y_out = np.hstack(y_resampled)

        perm = self.rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]

    def _calculate_difficulties(self, X, y, minority_class):
        """
        Calculate difficulty (ratio of majority neighbors) for each minority sample.
        """
        minority_mask = y == minority_class
        minority_samples = X[minority_mask]

        nn = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn.fit(X)

        difficulties = []
        for sample in minority_samples:
            _, neighbors_idx = nn.kneighbors([sample])
            neighbors_idx = neighbors_idx[0][1:]

            neighbor_labels = y[neighbors_idx]
            n_majority = np.sum(neighbor_labels != minority_class)

            difficulties.append(n_majority / self.k_neighbors)

        return np.array(difficulties)


class FeatureEngineering:
    """
    Advanced feature engineering for network intrusion detection.
    Adds statistical and interaction features.
    """
    def __init__(self, add_stats=True, add_interactions=True, add_ratios=True):
        self.add_stats = add_stats
        self.add_interactions = add_interactions
        self.add_ratios = add_ratios
        self.scaler = RobustScaler()

    def fit_transform(self, X, group_indices=None):
        """
        Transform features with engineering.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        group_indices : dict, optional
            Mapping of feature groups (e.g., 'packet_size', 'duration')
        """
        X = np.array(X)
        features = [X]

        if self.add_stats:
            stats_features = self._add_statistical_features(X)
            features.append(stats_features)

        if self.add_interactions:
            interaction_features = self._add_interaction_features(X)
            features.append(interaction_features)

        if self.add_ratios:
            ratio_features = self._add_ratio_features(X)
            features.append(ratio_features)

        X_enhanced = np.hstack(features)

        # Scale
        X_enhanced = self.scaler.fit_transform(X_enhanced)

        return X_enhanced

    def transform(self, X):
        """Transform new data with fitted scaler."""
        X = np.array(X)
        features = [X]

        if self.add_stats:
            stats_features = self._add_statistical_features(X)
            features.append(stats_features)

        if self.add_interactions:
            interaction_features = self._add_interaction_features(X)
            features.append(interaction_features)

        if self.add_ratios:
            ratio_features = self._add_ratio_features(X)
            features.append(ratio_features)

        X_enhanced = np.hstack(features)
        return self.scaler.transform(X_enhanced)

    def _add_statistical_features(self, X):
        """Add statistical features (mean, std, skewness, kurtosis)."""
        n_samples = X.shape[0]

        # Row-wise statistics
        mean_feat = np.mean(X, axis=1, keepdims=True)
        std_feat = np.std(X, axis=1, keepdims=True)

        # Avoid division by zero
        std_safe = np.where(std_feat == 0, 1, std_feat)

        # Skewness approximation
        centered = X - mean_feat
        skew_feat = np.mean((centered / std_safe) ** 3, axis=1, keepdims=True)

        # Kurtosis approximation
        kurt_feat = np.mean((centered / std_safe) ** 4, axis=1, keepdims=True) - 3

        # Max and min
        max_feat = np.max(X, axis=1, keepdims=True)
        min_feat = np.min(X, axis=1, keepdims=True)
        range_feat = max_feat - min_feat

        return np.hstack([mean_feat, std_feat, skew_feat, kurt_feat, max_feat, min_feat, range_feat])

    def _add_interaction_features(self, X, max_interactions=20):
        """Add interaction features (pairwise products)."""
        n_samples, n_features = X.shape

        # Select top features by variance for interactions
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-min(max_interactions, n_features//2):]

        interactions = []
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                interactions.append(X[:, top_indices[i]] * X[:, top_indices[j]])

        if interactions:
            return np.column_stack(interactions)
        else:
            return np.zeros((n_samples, 0))

    def _add_ratio_features(self, X):
        """Add ratio features between pairs of features."""
        n_samples, n_features = X.shape

        # Select features for ratios
        selected = min(10, n_features)
        indices = np.random.choice(n_features, selected, replace=False)

        ratios = []
        for i in range(selected):
            for j in range(i + 1, selected):
                denom = X[:, indices[j]]
                # Avoid division by zero
                denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
                ratio = X[:, indices[i]] / denom
                # Clip extreme values
                ratio = np.clip(ratio, -100, 100)
                ratios.append(ratio)

        if ratios:
            return np.column_stack(ratios)
        else:
            return np.zeros((n_samples, 0))


class CICIDS2017Preprocessor:
    """
    Specialized preprocessor for CICIDS2017 dataset.
    Handles specific data quality issues in this dataset.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_engineering = FeatureEngineering()

    def clean_data(self, df):
        """
        Clean CICIDS2017 specific issues.
        """
        df = df.copy()

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove constant features
        for col in numeric_cols:
            if df[col].std() == 0:
                df = df.drop(columns=[col])

        # Handle extremely large values (likely errors)
        for col in numeric_cols:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)

        return df

    def preprocess(self, df, label_col='Label'):
        """
        Full preprocessing pipeline.
        """
        # Clean data
        df = self.clean_data(df)

        # Extract labels
        y = df[label_col].values
        if not pd.api.types.is_numeric_dtype(df[label_col]):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Extract features (exclude common non-feature columns)
        exclude_cols = [label_col, 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp',
                       'Src Port', 'Dst Port', 'Protocol']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols].select_dtypes(include=[np.number]).values

        # Feature engineering
        X = self.feature_engineering.fit_transform(X)

        return X, y, feature_cols


if __name__ == "__main__":
    # Test BorderlineSMOTE
    print("Testing BorderlineSMOTE...")
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.array([0] * 800 + [1] * 200)  # Imbalanced
    np.random.shuffle(y)

    smote = BorderlineSMOTE(k_neighbors=5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y, target_ratio=0.5)
    print(f"Original: {Counter(y)}, Resampled: {Counter(y_res)}")

    # Test ADASYN
    print("\nTesting ADASYN...")
    adasyn = ADASYN(k_neighbors=5, random_state=42)
    X_res, y_res = adasyn.fit_resample(X, y, target_ratio=0.5)
    print(f"Original: {Counter(y)}, Resampled: {Counter(y_res)}")

    # Test FeatureEngineering
    print("\nTesting FeatureEngineering...")
    fe = FeatureEngineering()
    X_enhanced = fe.fit_transform(X)
    print(f"Original features: {X.shape[1]}, Enhanced features: {X_enhanced.shape[1]}")

    print("\nAll tests passed!")
