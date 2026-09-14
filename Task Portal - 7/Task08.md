# Task 8: Dimensionality Reduction

## Task Information

| Attribute | Details |
| --- | --- |
| **Task ID** | Task 8 |
| **Topic** | Dimensionality Reduction |
| **Domain** | Data Science & Machine Learning |
| **Deliverable** | Technical Documentation & Implementation (`.md`) |
| **Status** | Completed |

---

## Objective

To master the principles, mathematical foundations, and practical applications of Dimensionality Reduction techniques in Machine Learning—focusing on Feature Extraction (PCA, LDA, t-SNE) and Feature Selection to mitigate the Curse of Dimensionality while preserving critical data variance and class separability.

---

## Introduction & Conceptual Framework

High-dimensional feature spaces present significant challenges in machine learning, including severe computational latency, risk of overfitting, and data sparsity. This phenomenon is known as the **Curse of Dimensionality**.

Dimensionality Reduction reduces the number of input variables in a dataset by mapping high-dimensional data points to a lower-dimensional space while retaining maximum information.

```
High-Dimensional Input Data (D Features)
                  │
                  ├──► Feature Selection ──► Select Subset of Original Features (k < D)
                  │
                  └──► Feature Extraction ──► Construct New Latent Features (k < D)
                                               ├── Linear: PCA, LDA, Truncated SVD
                                               └── Non-Linear: t-SNE, UMAP, Kernel PCA

```

---

## Technical Content & Mathematical Foundations

### 1. Feature Selection vs. Feature Extraction

* **Feature Selection:** Subset selection without altering the original feature meanings.
* **Filter Methods:** ANOVA, Chi-Square, Pearson Correlation Coefficient.
* **Wrapper Methods:** Recursive Feature Elimination (RFE), Forward/Backward Selection.
* **Embedded Methods:** Lasso Regularization ($L_1$), Tree-based Feature Importance.


* **Feature Extraction:** Transforms the feature space to construct a new set of orthogonal or manifold-aligned latent variables.

---

### 2. Principal Component Analysis (PCA)

PCA is an unsupervised linear transformation technique that identifies orthogonal axes—termed **Principal Components**—along which the variance of the data is maximized.

#### Mathematical Steps:

1. **Data Standardization:**
Center the data to zero mean and unit variance for feature vector $\mathbf{x}$:

$$\mathbf{Z} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$


2. **Covariance Matrix Computation:**
Calculate the covariance matrix $\mathbf{\Sigma}$ of the standardized $n \times d$ data matrix $\mathbf{Z}$:

$$\mathbf{\Sigma} = \frac{1}{n - 1} \mathbf{Z}^T \mathbf{Z}$$


3. **Eigendecomposition:**
Solve for eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$:

$$\mathbf{\Sigma} \mathbf{v} = \lambda \mathbf{v}$$


4. **Explained Variance Ratio ($EVR$):**
The proportion of total variance captured by the $k$-th principal component:

$$EVR_k = \frac{\lambda_k}{\sum_{j=1}^{d} \lambda_j}$$



---

### 3. Linear Discriminant Analysis (LDA)

Unlike PCA, LDA is a supervised linear reduction technique designed to maximize class separability by projecting data onto a subspace that optimizes Fisher's criterion:

$$J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}$$

Where:

* $\mathbf{S}_B$ is the **Between-Class Scatter Matrix**.
* $\mathbf{S}_W$ is the **Within-Class Scatter Matrix**.

---

### 4. Non-Linear Techniques: t-SNE and UMAP

* **t-Distributed Stochastic Neighbor Embedding (t-SNE):** Converts Euclidean distances between data points into conditional probabilities that represent similarities. Excellent for 2D/3D visualization of high-dimensional clusters.
* **Uniform Manifold Approximation and Projection (UMAP):** Based on Riemannian geometry and algebraic topology. Superior to t-SNE at preserving both local and global data structures with significantly faster runtime.

---

## Code Implementation

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Create DataFrame of Results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['Target'] = y

print(f"Explained Variance Ratio by Component: {pca.explained_variance_ratio_}")
print(f"Total Preserved Variance: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

```

---

## Interview & Viva Questions

**Q1: Why is feature scaling mandatory prior to applying PCA?**
PCA maximizes variance along orthogonal directions. If features exist on vastly different scales (e.g., age in years vs. income in dollars), features with larger absolute ranges will artificially dominate the principal components regardless of their underlying informational value.

**Q2: How do you determine the optimal number of components in PCA?**
You evaluate a cumulative explained variance plot (scree plot) and select the number of components that reach a target threshold (typically 90%–95% of total variance) or identify the "elbow" point where adding more components yields diminishing returns.

**Q3: What are the main trade-offs between PCA and t-SNE?**
PCA is linear, computationally fast, deterministic, and preserves global variance structures, but struggles with complex non-linear manifolds. t-SNE excels at non-linear visualization and local cluster structure, but is computationally expensive, non-deterministic, and does not preserve global inter-cluster distances reliably.

---

Dimensionality reduction serves as a vital preprocessing and diagnostic step in data science pipelines. By stripping away redundant noise and collinearly dependent dimensions, models achieve improved computational efficiency, lower variance, and clearer visual interpretability.
