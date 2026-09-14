# Task 10 — Supervised Learning with Predictive Models: Regression, Classification, Ensemble Architectures, Loss Formulations & Evaluation Metrics

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VI |
| Task Number | 10 (Foundational Task) |
| Topic | Supervised Learning & Predictive Modeling: Parametric Regression (OLS, Ridge, Lasso, ElasticNet), Probabilistic Classification (Logistic Regression), Non-Linear Kernels (SVM), Tree Ensembles (Random Forest, GBDT, XGBoost, LightGBM, CatBoost), Bias-Variance Decomposition, and Model Evaluation Metrics |
| Task Type | Supervised Learning, Predictive Modeling & Regression/Classification Analysis |
| Status | Completed |
| Repository Section | `tasks/portal-06/task-04/` |

---

## 2. Objective

The objective of this task is to provide an in-depth mathematical, algorithmic, and practical exploration of **Supervised Predictive Modeling Algorithms** across continuous regression and discrete classification domains.
This task focuses on:
- Deriving **Parametric Linear Models** (Ordinary Least Squares, Ridge $L_2$, Lasso $L_1$, ElasticNet) and proving closed-form and iterative optimization limits.
- Formulating **Probabilistic Binary/Multiclass Classification** via Logistic Regression and Cross-Entropy loss functions.
- Analyzing **Support Vector Machines (SVM)** via primal/dual convex optimization, margin maximization, and non-linear kernel mappings.
- Formulating **Tree-Based Ensemble Methods**, contrasting Bagging (Random Forest) and Gradient Boosting (GBDT, XGBoost, LightGBM, CatBoost) via 2nd-order Taylor expansions.
- Rigorously deriving the **Bias-Variance Decomposition** to analyze model capacity, regularization constraints, and generalization bounds.
- Evaluating predictive performance using robust validation metrics (**Confusion Matrix**, **Precision**, **Recall**, **$F_1$-Score**, **ROC-AUC**, **PR-AUC**, **RMSE**, **MAE**, **$R^2$**, and **Adjusted $R^2$**).

---

## 3. Introduction

**Supervised Learning** is the predictive modeling paradigm where an algorithm learns a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from a labeled training dataset $\mathcal{D} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_N, y_N)\}$, where input feature vectors $\mathbf{x}_i \in \mathbb{R}^d$ correspond to target labels $y_i \in \mathcal{Y}$.

Depending on the target space $\mathcal{Y}$:
- **Continuous Target ($\mathcal{Y} = \mathbb{R}$):** Solved via **Regression Analysis** (e.g., predicting house prices, financial yields, sales volume).
- **Discrete Target ($\mathcal{Y} \in \{0, 1, \dots, C-1\}$):** Solved via **Classification Analysis** (e.g., credit default prediction, medical diagnosis, churn estimation).

```text
               Supervised Learning Pipeline Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ LABELED DATASET D = {(x1, y1), (x2, y2), ..., (xN, yN)}                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌─────────────────────────┐                           ┌───────────────────────┐
│ REGRESSION (Y ∈ ℝ)      │                           │ CLASSIFICATION (Y ∈ C)│
│ • OLS / Ridge / Lasso   │                           │ • Logistic Regression │
│ • Support Vector Reg    │                           │ • Support Vector Machine│
│ • GBDT / XGBoost Reg    │                           │ • Random Forest / GBDT│
└───────────┬─────────────┘                           └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EMPIRICAL RISK MINIMIZATION (ERM) & LOSS OPTIMIZATION                       │
│ • Loss Function: L(y, f(x; θ))                                              │
│ • Regularization: λ R(θ)                                                    │
│ • Objective: min_θ [ (1/N) ∑ L(yi, f(xi; θ)) + λ R(θ) ]                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL VALIDATION & OUT-OF-SAMPLE GENERALIZATION AUDIT                        │
│ • Classification: ROC-AUC, PR-AUC, Confusion Matrix, Log-Loss              │
│ • Regression: RMSE, MAE, R², Adjusted R²                                    │
└─────────────────────────────────────────────────────────────────────────────┘

```

The core principle governing supervised predictive modeling is:

> **Supervised predictive models construct explicit mapping functions from feature representations to target outputs by minimizing empirical risk over parameterized loss functions while constraining model complexity to ensure optimal generalization.**

---

## 4. Algorithmic Family Comparison Matrix

Choosing the right predictive algorithm depends on linearity assumptions, feature dimensions, dataset volume, outlier presence, and interpretability constraints.

```text
                     Supervised Model Taxonomy Matrix
┌─────────────┬──────────────────────────┬───────────────────────┬────────────┐
│ Model       │ Underlying               │ Geometric Boundary /  │ Parametric │
│ Family      │ Formulation              │ Structural Capability │ / Non-Par? │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Linear /    │ Linear combination of    │ Hyperplane decision   │ Parametric │
│ Regularized │ features; OLS / Logistic │ boundaries; assumes   │ (Fixed     │
│ Regression  │ maximum likelihood       │ linear feature spacing│ parameters)│
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Support     │ Margin maximization via  │ Linear hyperplanes or │ Hybrid     │
│ Vector      │ dual optimization &      │ non-linear manifolds  │ (Scales with│
│ Machines    │ kernel expansion         │ via kernel functions  │ Support Vec)│
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Decision    │ Recursive greedy space   │ Axis-aligned orthogonal│ Non-Param  │
│ Trees       │ partitioning via entropy │ rectangular feature   │ (Grows with│
│ (CART)      │ / Gini variance reduction│ hyper-rectangles      │ depth)     │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Ensemble    │ Bootstrap aggregation    │ Complex non-linear    │ Non-Param  │
│ (Random     │ (Bagging) or sequential  │ smooth boundaries;    │ (Robust    │
│ Forest/GBDT)│ gradient boosting        │ resists overfitting   │ non-linear)│
└─────────────┴──────────────────────────┴───────────────────────┴────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

Developing a rigorous understanding of supervised learning requires formalizing optimization objectives, convex dual problems, tree split criteria, loss derivatives, and variance decompositions.

### 5.1 Parametric Linear Models & Regularization

#### 1. Ordinary Least Squares (OLS) Regression:

Given design matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$ and target vector $\mathbf{y} \in \mathbb{R}^N$, OLS models predictions as $\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$. The Mean Squared Error (MSE) objective function is:

$$J(\boldsymbol{\beta}) = \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 = \frac{1}{2N} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Setting the gradient to zero ($\nabla_{\boldsymbol{\beta}} J(\boldsymbol{\beta}) = \mathbf{0}$) yields the **Normal Equation**:

$$\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y} \implies \hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

*Prerequisite:* $\mathbf{X}^T \mathbf{X}$ must be non-singular (invertible), requiring $N > d$ and no perfect multicollinearity.

#### 2. Regularized Regression Formulations:

To prevent overfitting and handle multicollinearity, regularization terms $R(\boldsymbol{\beta})$ are added:

* **Ridge Regression ($L_2$ Regularization):** Adds squared $L_2$ penalty. Constrains magnitude of parameters.

$$J_{\text{Ridge}}(\boldsymbol{\beta}) = \frac{1}{2N} \Vert{}\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\Vert{}_2^2 + \lambda \Vert{}\boldsymbol{\beta}\Vert{}_2^2 \implies \hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

* **Lasso Regression ($L_1$ Regularization):** Adds absolute $L_1$ penalty. Forces coefficients to exactly zero, performing continuous feature selection.

$$J_{\text{Lasso}}(\boldsymbol{\beta}) = \frac{1}{2N} \Vert{}\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\Vert{}_2^2 + \lambda \Vert{}\boldsymbol{\beta}\Vert{}_1$$

* **ElasticNet Regression:** Combines both $L_1$ and $L_2$ penalties with mixing parameter $\alpha \in [0, 1]$:

$$J_{\text{ElasticNet}}(\boldsymbol{\beta}) = \frac{1}{2N} \Vert{}\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\Vert{}_2^2 + \lambda \left[ \alpha \Vert{}\boldsymbol{\beta}\Vert{}_1 + \frac{1 - \alpha}{2} \Vert{}\boldsymbol{\beta}\Vert{}_2^2 \right]$$

#### 3. Logistic Regression (Probabilistic Binary Classification):

Logistic regression models the probability $P(y_i = 1 \mid \mathbf{x}_i)$ via the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$:

$$P(y_i = 1 \mid \mathbf{x}_i; \boldsymbol{\theta}) = h_{\boldsymbol{\theta}}(\mathbf{x}_i) = \sigma(\boldsymbol{\theta}^T \mathbf{x}_i) = \frac{1}{1 + e^{-\boldsymbol{\theta}^T \mathbf{x}_i}}$$

Parameters are estimated by maximizing log-likelihood, equivalent to minimizing **Binary Cross-Entropy (Log Loss)**:

$$J(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(h_{\boldsymbol{\theta}}(\mathbf{x}_i)) + (1 - y_i) \log(1 - h_{\boldsymbol{\theta}}(\mathbf{x}_i)) \right]$$

Gradient update step via Gradient Descent:

$$\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \frac{1}{N} \mathbf{X}^T (h_{\boldsymbol{\theta}}(\mathbf{X}) - \mathbf{y})$$

---

### 5.2 Non-Parametric Kernel Models (Support Vector Machines)

Support Vector Machines construct a hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ that maximizes the geometric margin between binary classes $y_i \in \{-1, +1\}$.

```text
                     SVM Support Vector Margin Geometry
                     
                     Hyperplane: w^T x + b = +1 (Upper Boundary)
                     ----------------------------------------- ● (Support Vector)
                                    ↑
                        Margin γ    │  Separating Hyperplane: w^T x + b = 0
                        γ = 2 / ‖w‖ │  -----------------------------------
                                    ↓
                     ----------------------------------------- ○ (Support Vector)
                     Hyperplane: w^T x + b = -1 (Lower Boundary)

```

#### 1. Soft-Margin Primal Optimization Problem:

Allows for misclassified points using slack variables $\xi_i \ge 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \Vert{}\mathbf{w}\Vert{}_2^2 + C \sum_{i=1}^N \xi_i$$

$$\text{Subject to: } y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0 \quad \forall i$$

Where $C > 0$ balances margin maximization and classification error.

#### 2. Dual Formulation & Kernel Trick:

Using Lagrange multipliers $\alpha_i \ge 0$, the dual optimization problem becomes:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

$$\text{Subject to: } 0 \le \alpha_i \le C \quad \text{and} \quad \sum_{i=1}^N \alpha_i y_i = 0$$

The **Kernel Trick** replaces the inner product $\langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$:

* **Radial Basis Function (RBF / Gaussian) Kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \Vert{}\mathbf{x}_i - \mathbf{x}_j\Vert{}_2^2 \right)$
* **Polynomial Kernel:** $K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j + c)^d$

#### 3. Equivalent Hinge Loss Formulation:

$$J_{\text{SVM}}(\mathbf{w}, b) = \frac{1}{2} \Vert{}\mathbf{w}\Vert{}_2^2 + C \sum_{i=1}^N \max\left(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)\right)$$

---

### 5.3 Tree-Based Ensemble Methods

#### 1. Decision Tree Split Impurity Criteria:

Decision Trees partition feature space recursively. At node $m$ with dataset $D_m$:

* **Gini Impurity (Classification):** Measures probability of mislabeling a randomly chosen element.

$$G(D_m) = 1 - \sum_{k=1}^C p_k^2, \quad \text{where } p_k = \frac{1}{\vert{}D_m\vert{}} \sum_{i \in D_m} \mathbb{I}(y_i = k)$$

* **Cross-Entropy (Classification):**

$$H(D_m) = -\sum_{k=1}^C p_k \log_2(p_k)$$

* **Variance Reduction (Regression):**

$$\text{Var}(D_m) = \frac{1}{\vert{}D_m\vert{}} \sum_{i \in D_m} (y_i - \bar{y}_m)^2$$

A split on feature $j$ at threshold $t$ maximizes **Information Gain**:

$$\text{Gain}(D_m, j, t) = I(D_m) - \left( \frac{\vert{}D_L\vert{}}{\vert{}D_m\vert{}} I(D_L) + \frac{\vert{}D_R\vert{}}{\vert{}D_m\vert{}} I(D_R) \right)$$

#### 2. Random Forests (Bagging + Feature Subspace):

Combines two variance-reduction mechanisms:

* **Bootstrap Aggregating (Bagging):** Trains $M$ independent decision trees on $M$ bootstrap samples drawn with replacement from $\mathcal{D}$.
* **Random Subspace Method:** Selects a random subset of $m = \sqrt{d}$ (for classification) or $m = d/3$ (for regression) features at each split.
Final prediction for classification: $\hat{y} = \text{mode}(\{\hat{y}_1, \dots, \hat{y}_M\})$.

#### 3. Gradient Boosted Decision Trees (GBDT):

GBDT builds trees sequentially in a stage-wise additive manner: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})$.

At iteration $m$, tree $h_m(\mathbf{x})$ is fit to the **pseudo-residuals** (negative gradients of loss $\mathcal{L}(y, F(\mathbf{x}))$):

$$r_{i,m} = -\left[ \frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F(\mathbf{x}) = F_{m-1}(\mathbf{x})}$$

#### 4. XGBoost Optimization (Second-Order Taylor Expansion):

XGBoost optimizes objective function at step $m$ using second-order Taylor expansion:

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ \mathcal{L}(y_i, F_{m-1}(\mathbf{x}_i)) + g_i h_m(\mathbf{x}_i) + \frac{1}{2} h_i h_m^2(\mathbf{x}_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$

Where first and second derivatives are:

$$g_i = \frac{\partial \mathcal{L}(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}, \quad h_i = \frac{\partial^2 \mathcal{L}(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}^2(\mathbf{x}_i)}$$

Optimal leaf weight $w_j^*$ for leaf $j$ containing sample set $I_j$:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

---

### 5.4 The Bias-Variance Decomposition

For a regression model $\hat{f}(\mathbf{x})$ trained on dataset $\mathcal{D}$ generated from ground truth $y = f(\mathbf{x}) + \epsilon$ (where noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$), expected out-of-sample Mean Squared Error decomposes into three distinct components:

$$\mathbb{E}_{\mathcal{D}}\left[ (y - \hat{f}(\mathbf{x}))^2 \right] = \text{Bias}\left[\hat{f}(\mathbf{x})\right]^2 + \text{Variance}\left[\hat{f}(\mathbf{x})\right] + \sigma^2$$

Where:

* **$\text{Bias}\left[\hat{f}(\mathbf{x})\right] = \mathbb{E}_{\mathcal{D}}\left[\hat{f}(\mathbf{x})\right] - f(\mathbf{x})$:** Error introduced by approximating a real-world problem with a simplified model (Underfitting).
* **$\text{Variance}\left[\hat{f}(\mathbf{x})\right] = \mathbb{E}_{\mathcal{D}}\left[ \left( \hat{f}(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}\left[\hat{f}(\mathbf{x})\right] \right)^2 \right]$:** Model sensitivity to small fluctuations in training data (Overfitting).
* **$\sigma^2$:** Irreducible error inherent to the noise in the underlying data generating process.

```text
                     The Bias-Variance Tradeoff Curves
      Error
        ▲
        │    Total Error = Bias² + Variance + Noise
        │    \                               /
        │     \    Optimal Model            /   High Variance
        │      \   Complexity              /   (Overfitting)
        │       \      │                  /
        │        \     │                 /     Variance
        │  High Bias    \               /     ── ── ── ──
        │ (Underfitting) \_           _/
        │                  \_________/
        │    Bias²
        │    ─────── ── ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        │    Irreducible Noise (σ²)
        └──────────────────────────────────────────────────► Model Complexity

```

---

### 5.5 Evaluation Metrics & Model Validation

#### Classification Metrics:

1. **Confusion Matrix Components:** True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), False Negatives ($FN$).
2. **Precision & Recall:**

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

3. **$F_\beta$-Score:** Weighted harmonic mean balancing Precision and Recall.

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

For $\beta = 1$, $F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$.

4. **ROC-AUC Score:** Area under Receiver Operating Characteristic curve plotting True Positive Rate ($TPR = \frac{TP}{TP+FN}$) versus False Positive Rate ($FPR = \frac{FP}{FP+TN}$) across all classification thresholds $\tau \in [0, 1]$.

#### Regression Metrics:

1. **Root Mean Squared Error (RMSE):** Penalizes larger errors heavily.

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

2. **Mean Absolute Error (MAE):** Robust metric measuring average absolute deviation.

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N \vert{}y_i - \hat{y}_i\vert{}$$

3. **Coefficient of Determination ($R^2$) & Adjusted $R^2$:**
Proportion of variance explained by model relative to baseline mean model:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

$$\text{Adjusted } R^2 = 1 - \left[ (1 - R^2) \frac{N - 1}{N - d - 1} \right]$$

Where $d$ is the number of features. Adjusted $R^2$ penalizes adding features that do not significantly improve model fit.

---

## 6. Enterprise Predictive Modeling Architecture

Production machine learning pipelines isolate feature engineering, hyperparameter search, ensemble blending, and model auditing into modular deployment layers.

```text
            Production Predictive Inference & Training Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ INGESTION & FEATURE ENGINEERING ENGINE                                      │
│ • One-Hot / Target Encoding, Scalers, Imputers & Polynomial Features        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STRATIFIED K-FOLD CV & HYPERPARAMETER SEARCH GATEWAY                        │
│ • Bayesian Optimization (Optuna) searching learning rate, depth, l2 penalty │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ENSEMBLE STACKING & BLENDING LAYER                                          │
│ ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│ │ Model 1: LightGBM    │  │ Model 2: XGBoost     │  │ Model 3: CatBoost    │ │
│ └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘ │
│            └─────────────────────────┼─────────────────────────┘             │
│                                      ▼                                       │
│                    Meta-Learner: Logistic/Ridge Regression                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CALIBRATION, DRIFT MONITORS & MODEL REGISTRY EXPORT                         │
│ • Platt Scaling / Isotonic Calibration to ensure valid probabilities        │
│ • Export to ONNX / MLflow Model Registry for real-time scoring API          │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Metric & Selection Matrix

| Metric | Task Domain | Value Range | Optimal Target | Primary Characteristics & Best Use Case |
| --- | --- | --- | --- | --- |
| **$F_1$-Score** | Classification | $[0, 1]$ | $1.0$ | Harmonic mean of precision and recall. Best for balanced evaluation on imbalanced datasets. |
| **ROC-AUC** | Classification | $[0, 1]$ | $1.0$ | Threshold-invariant measure of ranking quality. Ideal for balanced classes and general scoring capability. |
| **PR-AUC** | Classification | $[0, 1]$ | $1.0$ | Focuses exclusively on the positive class. **Best evaluation metric for severely imbalanced data**. |
| **Log-Loss** | Classification | $[0, \infty)$ | $0.0$ | Penalizes confident incorrect probabilistic predictions heavily. Essential for probability calibration. |
| **RMSE** | Regression | $[0, \infty)$ | $0.0$ | Quadratic loss weighting. Sensitive to extreme outliers in continuous regression targets. |
| **MAE** | Regression | $[0, \infty)$ | $0.0$ | Linear loss weighting. Robust to target outliers; provides intuitive error magnitude interpretation. |
| **Adjusted $R^2$** | Regression | $(-\infty, 1]$ | $1.0$ | Quantifies explained variance while penalizing redundant parameters. Preferred for multiple linear regression. |

---

## 8. Technology & Implementation Matrix

| Framework / Tool | Core Algorithms & Modules | Key Optimization Parameters | Production Recommendation |
| --- | --- | --- | --- |
| **Scikit-Learn** | `LinearRegression`, `LogisticRegression`, `SVM`, `RandomForestClassifier` | `C`, `penalty`, `alpha`, `n_estimators`, `max_depth` | Industry standard baseline for classical machine learning and baseline pipelines. |
| **XGBoost (`xgboost`)** | `XGBClassifier`, `XGBRegressor` | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma` | Production framework for high-speed gradient boosting on structured tabular data. |
| **LightGBM (`lightgbm`)** | `LGBMClassifier`, `LGBMRegressor` | `num_leaves`, `max_depth`, `learning_rate`, `min_child_samples` | Optimized for ultra-large tabular datasets; uses leaf-wise tree growth. |
| **CatBoost (`catboost`)** | `CatBoostClassifier`, `CatBoostRegressor` | `depth`, `learning_rate`, `l2_leaf_reg`, `cat_features` | Specialized framework for tabular data with high-cardinality categorical features. |
| **Optuna** | `optuna.create_study()`, `TPESampler` | `n_trials`, `timeout`, hyperparameter search space | Automated Bayesian hyperparameter optimization framework. |

---

## 9. Personal Understanding

Task 04 clarifies how supervised learning algorithms map input features to discrete or continuous targets.

Key personal insights include:

1. **No single algorithm fits all data structures:** Linear models (Ridge, Lasso) work well when features have strong linear relationships with the target and $N < d$. Tree ensembles (Random Forest, XGBoost) excel on non-linear tabular data with interacting features, but cannot extrapolate beyond the min/max bounds of the training targets.
2. **Regularization directly controls the Bias-Variance Tradeoff:** Unregularized models with high capacity (e.g., deep decision trees, high $C$ in SVMs) fit training noise, leading to low bias but high variance (overfitting). Adding regularizers ($L_1$/$L_2$ penalties, tree depth limits, learning rate shrinkage) increases bias slightly while substantially decreasing variance.
3. **Probability Calibration is critical for enterprise decision-making:** Many high-performing classifiers (e.g., SVMs, boosting models with deep trees) produce uncalibrated scores that do not reflect true posterior probabilities. Applying post-processing techniques like **Platt Scaling** or **Isotonic Regression** ensures predicted probabilities align with real-world observed event frequencies.

The core principle remains:

> **Supervised predictive models construct explicit mapping functions from feature representations to target outputs by minimizing empirical risk over parameterized loss functions while constraining model complexity to ensure optimal generalization.**

---

## 10. Interview / Viva Questions

### Q1. Derive the closed-form Normal Equation solution for Ordinary Least Squares (OLS) regression.

**Answer:**

Given loss function $J(\boldsymbol{\beta}) = \frac{1}{2} (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \frac{1}{2} \left( \mathbf{y}^T \mathbf{y} - 2\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} \right)$.

Taking derivative with respect to $\boldsymbol{\beta}$ and setting to zero:

$$\frac{\partial J(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = -\mathbf{X}^T \mathbf{y} + \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{0} \implies \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y} \implies \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Q2. Why is Mean Squared Error (MSE) unsuitable as a loss function for Logistic Regression?

**Answer:**

Using MSE with the non-linear sigmoid activation function $h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^T \mathbf{x})$ produces a non-convex loss surface with many local minima. Gradient descent can get trapped in these local minima. Binary Cross-Entropy (Log-Loss) guarantees a strictly **convex** loss function with a unique global minimum.

### Q3. Compare Ridge ($L_2$) and Lasso ($L_1$) regularization. Why does Lasso produce sparse coefficients?

**Answer:**

* **Ridge ($L_2$):** Penalizes squared weight magnitudes ($\lambda \sum \beta_j^2$). It shrinks weights smoothly toward zero but never makes them exactly zero.
* **Lasso ($L_1$):** Penalizes absolute weight magnitudes ($\lambda \sum \vert{}\beta_j\vert{}$). Geometrically, the $L_1$ constraint region is a diamond with sharp corners at the coordinate axes. The elliptical level sets of the unconstrained OLS loss function often intersect the $L_1$ constraint region at these corners, setting coefficient values to exactly zero and performing feature selection.

```text
               Geometric Constraint Contours: L1 vs L2
       L1 (Lasso - Diamond)                   L2 (Ridge - Circle)
             β2 ▲                                   β2 ▲
                │   / OLS Contours                     │   / OLS Contours
               /│\ /                                  /│\ /
              / │ \                                  / │ \
       ───────●─┼───●──────► β1               ──────(─┼─)──────► β1
              \ │ /                                  \ │ /
               \│/                                    \│/
                │                                      │

```

### Q4. Derive the mathematical Bias-Variance Decomposition of Mean Squared Error.

**Answer:**

Let $y = f(\mathbf{x}) + \epsilon$ where $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$. Let $\hat{f} = \hat{f}(\mathbf{x})$ be the model prediction.

$$\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[(f + \epsilon - \hat{f})^2] = \mathbb{E}[( (f - \hat{f}) + \epsilon )^2]$$

$$\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[(f - \hat{f})^2] + 2\mathbb{E}[(f - \hat{f})\epsilon] + \mathbb{E}[\epsilon^2]$$

Since noise $\epsilon$ is independent of data and model, $\mathbb{E}[(f - \hat{f})\epsilon] = 0$, so:

$$\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[(f - \hat{f})^2] + \sigma^2$$

Adding and subtracting $\mathbb{E}[\hat{f}]$ inside the squared term:

$$\mathbb{E}[(f - \hat{f})^2] = \mathbb{E}\left[ \left( (f - \mathbb{E}[\hat{f}]) + (\mathbb{E}[\hat{f}] - \hat{f}) \right)^2 \right]$$

$$\mathbb{E}[(f - \hat{f})^2] = (f - \mathbb{E}[\hat{f}])^2 + \mathbb{E}\left[(\hat{f} - \mathbb{E}[\hat{f}])^2\right] = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f})$$

Combining terms yields the final decomposition:

$$\mathbb{E}[(y - \hat{f})^2] = \text{Bias}(\hat{f})^2 + \text{Var}(\hat{f}) + \sigma^2$$

### Q5. What is the Kernel Trick in Support Vector Machines, and how does it lower computational cost?

**Answer:**

The **Kernel Trick** calculates the inner product of vectors in a high-dimensional feature space $\mathcal{H}$ directly in the lower-dimensional input space using a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$. This avoids explicitly transforming vectors into high-dimensional space $\Phi(\mathbf{x})$, reducing computational complexity from $\mathcal{O}(\dim(\mathcal{H}))$ to $\mathcal{O}(d)$.

### Q6. Contrast Bagging (Random Forest) and Gradient Boosting (GBDT).

**Answer:**

* **Bagging (Random Forest):** Trains multiple deep decision trees **in parallel** on independent bootstrap samples. Reduces overall model **variance** without increasing bias.
* **Boosting (GBDT):** Trains multiple shallow decision trees **sequentially**, with each new tree fitting the residual errors of the prior ensemble. Reduces overall model **bias** while controlling variance via learning rate shrinkage ($\eta$).

### Q7. How does XGBoost use second-order Taylor expansions to optimize custom loss functions?

**Answer:**

Traditional gradient boosting uses first-order gradients (steepest descent). XGBoost expands the loss function using a 2nd-order Taylor series around step $m-1$:

$$\mathcal{L}^{(m)} \approx \sum_{i=1}^N \left[ \mathcal{L}(y_i, F_{m-1}(\mathbf{x}_i)) + g_i f_m(\mathbf{x}_i) + \frac{1}{2} h_i f_m^2(\mathbf{x}_i) \right] + \Omega(f_m)$$

Where $g_i = \frac{\partial \mathcal{L}}{\partial F_{m-1}}$ (1st derivative) and $h_i = \frac{\partial^2 \mathcal{L}}{\partial F_{m-1}^2}$ (2nd derivative / Hessian). This second-order approximation allows XGBoost to optimize any custom differentiable loss function efficiently.

### Q8. Compare Gini Impurity and Information Entropy for Decision Tree splitting.

**Answer:**

* **Gini Impurity:** $G = 1 - \sum_{k=1}^C p_k^2$. Measures the variance of categorical distributions. Computationally faster because it avoids logarithmic calculations.
* **Entropy:** $H = -\sum_{k=1}^C p_k \log_2(p_k)$. Originates from information theory. Penalizes mixed class distributions slightly more heavily, producing slightly more balanced tree splits.

### Q9. How do you handle severe class imbalance ($>99:1$) in supervised classification?

**Answer:**

1. **Resampling:** Use SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN to oversample minority classes, or use Random Under-Sampling (RUS).
2. **Cost-Sensitive Learning:** Adjust class weights in the loss function (e.g., `scale_pos_weight` in XGBoost, `class_weight='balanced'` in Scikit-Learn).
3. **Metric Selection:** Evaluate models using **PR-AUC** and **$F_\beta$-score** instead of standard accuracy or ROC-AUC.

### Q10. Compare the Receiver Operating Characteristic (ROC) curve and the Precision-Recall (PR) curve.

**Answer:**

* **ROC Curve:** Plots True Positive Rate ($TPR$) vs. False Positive Rate ($FPR$). It is invariant to class distribution changes, making it useful when evaluating general ranking ability on balanced datasets.
* **PR Curve:** Plots Precision vs. Recall. It excludes True Negatives ($TN$), making it sensitive to false positive spikes in imbalanced datasets. It is preferred when minority class detection is critical.

### Q11. What is Out-of-Bag (OOB) error in Random Forests, and why does it eliminate the need for a separate validation set?

**Answer:**

When bootstrapping $N$ samples with replacement, the probability that a specific sample is not chosen is $\left(1 - \frac{1}{N}\right)^N \approx e^{-1} \approx 0.368$. Approximately $36.8\%$ of data points are excluded from each tree's training set (the Out-of-Bag samples). Aggregating predictions for each instance using only the trees where that instance was OOB yields an unbiased out-of-sample error estimate without holding out a separate validation set.

### Q12. Why is Adjusted $R^2$ preferred over standard $R^2$ in multiple linear regression?

**Answer:**

Standard $R^2$ always increases or remains constant when new features are added to a regression model, even if those features add no predictive value. **Adjusted $R^2$** adds a penalty for the number of features $d$:

$$\text{Adjusted } R^2 = 1 - \left[ (1 - R^2) \frac{N - 1}{N - d - 1} \right]$$

If a newly added feature does not reduce residual variance enough to offset the increase in $d$, Adjusted $R^2$ decreases.

### Q13. How does LightGBM's Leaf-Wise tree growth differ from XGBoost's Depth-Wise tree growth?

**Answer:**

* **Depth-Wise (Level-Wise):** Splits nodes level by level across the entire tree depth, maintaining a balanced tree structure. It limits overfitting but can waste computation on nodes with low loss reduction.
* **Leaf-Wise (Best-First):** Splits the single node that yields the maximum loss reduction regardless of level. This reduces loss faster, but can cause overfitting on smaller datasets unless constrained by `max_depth` or `num_leaves`.

```text
       Depth-Wise Tree Growth                 Leaf-Wise Tree Growth
              [Root]                                  [Root]
             /      \                                /      \
          [Node]   [Node]                         [Node]   [Node]
          /    \   /    \                                 /    \
        [L]    [L][L]   [L]                             [L]    [Node]
                                                               /    \
                                                             [L]    [L]

```

### Q14. What is model probability calibration, and why is an uncalibrated high ROC-AUC classifier dangerous in production?

**Answer:**

A classifier is **calibrated** if a predicted score of $0.80$ means the event occurs $80\%$ of the time in practice. Tree boosting and SVM models often produce distorted, uncalibrated score distributions even when their ROC-AUC score is high (since ROC-AUC measures ranking order, not score scale). In enterprise risk or medical settings, uncalibrated scores lead to bad thresholding decisions and inaccurate risk assessments.

### Q15. How does CatBoost process high-cardinality categorical features without one-hot encoding?

**Answer:**

CatBoost uses **Ordered Target Statistics**. For a categorical feature, it calculates target encoding values using only historical samples that appear before the current instance in a random permutation of the data. This prevents target leakage and overfitting:

$$\hat{x}_{i, k} = \frac{\sum_{j \in \text{Permutation prior}} \mathbb{I}(x_{j, k} = x_{i, k}) y_j + a \cdot P}{\sum_{j \in \text{Permutation prior}} \mathbb{I}(x_{j, k} = x_{i, k}) + a}$$

Where $P$ is a prior value and $a$ is a smoothing parameter.

---

## 11. Conclusion

Task 04 provides a thorough analysis of Supervised Learning Algorithms, Loss Optimization, and Evaluation Metrics.

```text
                 Supervised Predictive Modeling Execution Flow
                                       ↓
Dataset Ingestion, Imputation, & Feature Transformation (Scalers, Encoders)
                                       ↓
Stratified Cross-Validation Split & Baseline Selection (Linear/Logistic vs Tree Ensembles)
                                       ↓
Loss Function Optimization & Regularization Tuning (L1/L2, Tree Depth, Shrinkage)
                                       ↓
Model Evaluation & Metric Verification (ROC-AUC, PR-AUC, RMSE, MAE, R²)
                                       ↓
Probability Calibration & Model Deployment Export

```

The core structural pillars of Supervised Predictive Learning include:

```text
Supervised Predictive Learning Pillars
├── Parametric & Regularized Regression (OLS, Normal Eq, Ridge L2, Lasso L1, ElasticNet)
├── Probabilistic & Margin Boundaries (Logistic Regression, Soft-Margin SVM, Kernel Trick)
├── Non-Parametric Ensemble Architectures (Random Forest Bagging, GBDT, XGBoost, LightGBM)
└── Bias-Variance Mechanics & Metrics (MSE Decomposition, PR-AUC, Log-Loss, Adjusted R²)

```

Core tools and operational frameworks:

```text
Scikit-Learn (Classical estimators, cross-validation metrics, preprocessing scalers)
XGBoost & LightGBM (Gradient boosted decision trees for tabular datasets)
CatBoost Framework (Optimized handling for high-cardinality categorical features)
Optuna Engine (Bayesian hyperparameter optimization & automated trial sampling)

```

Completing Task 04 provides the core understanding required to build production supervised machine learning models, select appropriate loss functions, prevent overfitting, and evaluate performance accurately.

The central principle remains:

> **Supervised predictive models construct explicit mapping functions from feature representations to target outputs by minimizing empirical risk over parameterized loss functions while constraining model complexity to ensure optimal generalization.**

---

## 12. Key Takeaways

1. **Supervised Learning** trains models to map input features $\mathbf{x}$ to continuous or discrete targets $y$ using empirical risk minimization.
2. **OLS Regression** minimizes squared errors, yielding the closed-form Normal Equation solution $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.
3. **Ridge Regression ($L_2$)** shrinks feature weights smoothy to reduce variance, while **Lasso ($L_1$)** forces weights to zero to perform feature selection.
4. **Logistic Regression** models log-odds using the sigmoid function $\sigma(z)$, optimizing Binary Cross-Entropy (Log-Loss) via gradient descent.
5. **Support Vector Machines (SVM)** maximize the margin between classes; kernel functions ($K(\mathbf{x}_i, \mathbf{x}_j)$) enable non-linear classification without expanding features explicitly.
6. **Decision Trees** split feature space greedily by maximizing Information Gain (Gini Impurity, Entropy, or Variance Reduction).
7. **Random Forests** reduce variance by combining Bootstrap Aggregating (Bagging) with Random Subspace feature sampling across deep trees.
8. **Gradient Boosted Decision Trees (GBDT)** build shallow trees sequentially to fit the negative gradients (pseudo-residuals) of a loss function, reducing model bias.
9. **XGBoost** optimizes regularized objective functions using second-order Taylor expansions ($g_i, h_i$) for fast, accurate splits.
10. **LightGBM** uses leaf-wise tree growth and histogram-based binning to train quickly on large tabular datasets.
11. **CatBoost** uses ordered target statistics to handle categorical features directly without one-hot encoding.
12. **Bias-Variance Decomposition** shows that total expected error equals $\text{Bias}^2 + \text{Variance} + \sigma^2$ (irreducible noise).
13. **PR-AUC** is preferred over ROC-AUC for evaluating model performance under severe class imbalance.
14. **Adjusted $R^2$** evaluates regression fit while penalizing redundant features, preventing inflated accuracy estimates.
15. **Probability Calibration** (e.g., Platt Scaling) ensures predicted probabilities align with real-world observed event frequencies.
