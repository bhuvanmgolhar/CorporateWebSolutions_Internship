# Task 09 — Anomaly Detection: Statistical Approaches, Distance & Density-Based Methods, Isolation Trees, Reconstruction Autoencoders & Evaluation Metrics

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VI |
| Task Number | 09 (Foundational Task) |
| Topic | Anomaly & Outlier Detection: Statistical Tests (Z-Score, IQR, Mahalanobis), Density/Distance Methods (LOF, $k$-NN), Tree Isolation (Isolation Forest, EIF), Support Vector Boundaries (OC-SVM, SVDD), Deep Learning Reconstruction (Autoencoders, VAEs), and Imbalanced Evaluation Metrics |
| Task Type | Unsupervised & Semi-Supervised Learning, Outlier Detection & Anomaly Engineering |
| Status | Completed |
| Repository Section | `tasks/portal-06/task-09/` |

---

## 2. Objective

The objective of this task is to provide an in-depth mathematical, algorithmic, and practical exploration of **Anomaly and Outlier Detection Methods** in machine learning and spatial data analysis.
This task focuses on:
- Formalizing the taxonomy of anomalies (**Point**, **Contextual**, and **Collective** anomalies) across unsupervised, semi-supervised, and supervised settings.
- Deriving **Statistical Anomaly Detection** techniques (Z-Score, Median Absolute Deviation, Interquartile Range, Grubbs' Test, and Mahalanobis Distance).
- Analyzing **Distance & Density-Based Detection** algorithms (k-Nearest Neighbors Distance, Local Outlier Factor — LOF).
- Formulating **Partition & Isolation-Based** algorithms (Isolation Forest, Extended Isolation Forest) and proving path length convergence.
- Analyzing **One-Class Boundary Estimators** (One-Class SVM, Support Vector Data Description — SVDD) via kernel trick optimization.
- Developing **Reconstruction-Based Deep Architectures** (Autoencoders, Variational Autoencoders) using reconstruction error thresholds.
- Rigorously evaluating performance using imbalanced metrics (**ROC-AUC**, **PR-AUC**, **Precision@K**, **Adjusted F1 score**) under extreme class imbalance.

---

## 3. Introduction

**Anomaly Detection** (or Outlier Detection) is the process of identifying data points, items, or events $\mathbf{x}_i \in \mathbb{R}^d$ that deviate significantly from the expected behavior or distribution of a baseline dataset $\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$. Anomalies frequently represent critical events, such as financial fraud, network intrusion, sensor failures, structural defects, or rare medical conditions.

```text
                 Anomaly Detection Taxonomy & Structural Types
┌─────────────────────────────────────────────────────────────────────────────┐
│ UNLABELED / MULTIVARIATE DATASET X ∈ ℝ^(N × d)                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ POINT ANOMALIES      │   │ CONTEXTUAL ANOMALIES │   │ COLLECTIVE ANOMALIES │
│ Individual data      │   │ Normal in general,   │   │ Individual points    │
│ points lying far     │   │ anomalous within a   │   │ appear normal, but   │
│ outside feature space│   │ specific context     │   │ collective sequence  │
│ boundaries.          │   │ (e.g., temporal/geo).│   │ indicates anomaly.   │
└───────────┬──────────┘   └───────────┬──────────┘   └───────────┬──────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ANOMALY SCORING & THRESHOLDING ENGINE                                       │
│ • Score Assignment: S(x) ∈ [0, 1] or S(x) ∈ (-∞, +∞)                       │
│ • Decision Gate: S(x) ≥ τ ──► Flag as Anomaly | S(x) < τ ──► Normal         │
└─────────────────────────────────────────────────────────────────────────────┘

```

The core principle governing anomaly detection is:

> **Anomaly detection algorithms isolate rare pattern occurrences by defining normal structural distributions, measuring local or global spatial density deviations, or computing reconstruction boundaries within feature representations.**

---

## 4. Algorithmic Family Comparison Matrix

Choosing the right anomaly detection methodology depends on dataset volume, feature dimensionality, non-linear dependencies, and whether anomalous labels are present.

```text
                     Anomaly Detection Algorithm Matrix
┌─────────────┬──────────────────────────┬───────────────────────┬────────────┐
│ Algorithm   │ Underlying Model         │ Primary Assumption /  │ Dimensional│
│ Family      │ Formulation              │ Geometric Capability  │ Scalability│
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Statistical │ Parameter estimation     │ Data follows a known  │ Low        │
│ (Z-Score/   │ (Gaussian, Covariance    │ parametric standard   │ (Collapses │
│ Mahalanobis)│ Ellipsoid matrices)      │ probability density.  │ on d > 20) │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Density     │ Local Reachability       │ Anomalies reside in   │ Medium     │
│ (LOF / k-NN)│ Density ratios           │ low-density spatial   │ (Slow on   │
│             │ against k-neighbors      │ local neighborhoods.  │ N > 100k)  │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Isolation   │ Random feature splitting │ Anomalies are easy to │ High       │
│ (Isolation  │ trees (Path length       │ isolate, yielding     │ (Fast on   │
│ Forest)     │ calculation)             │ shorter tree paths.   │ high N, d) │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Boundary    │ Kernel hyper-plane /     │ Normal points lie     │ Medium     │
│ (One-Class  │ minimal hyper-sphere     │ within a tight kernel │ (Quadratic │
│ SVM / SVDD) │ boundary estimation      │ convex envelope.      │ in N)      │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Deep Learning│ Latent compression &     │ Anomalies cannot be   │ Very High  │
│ (Autoencoder│ reconstruction error     │ reconstructed well by │ (Scales to │
│ / VAE)      │ optimization             │ normal latent code.   │ images/text)│
└─────────────┴──────────────────────────┴───────────────────────┴────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

A complete understanding of anomaly detection requires formalizing parametric statistical bounds, density reachability, isolation trees, optimization loss functions, and evaluation metrics.

### 5.1 Statistical & Parametric Anomaly Detection

#### 1. Z-Score and Modified Z-Score (Median Absolute Deviation):

For a univariate Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, the standard Z-Score for observation $x_i$ is:

$$Z_i = \frac{x_i - \mu}{\sigma}$$

An observation is flagged as an anomaly if $\vert{}Z_i\vert{} > 3$ ($99.73\%$ confidence bound).

Because mean $\mu$ and standard deviation $\sigma$ are sensitive to extreme outliers, the **Modified Z-Score** uses the median $\tilde{x}$ and **Median Absolute Deviation (MAD)**:

$$\text{MAD} = \text{median}\left( \vert{}x_i - \tilde{x}\vert{} \right)$$

$$M_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{\text{MAD}}$$

Where $0.6745$ aligns the score with the standard normal distribution interquartile range. Points with $\vert{}M_i\vert{} > 3.5$ are identified as anomalies.

#### 2. Interquartile Range (IQR) Rule:

Defines non-parametric boundary limits based on percentiles ($Q_1 = 25\text{th}$, $Q_3 = 75\text{th}$):

$$\text{IQR} = Q_3 - Q_1$$

$$\text{Lower Bound} = Q_1 - 1.5 \cdot \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \cdot \text{IQR}$$

#### 3. Mahalanobis Distance (Multivariate Outlier Detection):

Measures the distance of a multivariate point $\mathbf{x}_i \in \mathbb{R}^d$ from the sample mean vector $\boldsymbol{\mu}$, taking covariance structure $\mathbf{\Sigma}$ into account:

$$D_M(\mathbf{x}_i) = \sqrt{(\mathbf{x}_i - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})}$$

Under multivariate normality, $D_M^2(\mathbf{x}_i)$ follows a Chi-Square distribution $\chi_d^2$ with $d$ degrees of freedom. Points where $D_M^2(\mathbf{x}_i) > \chi_{d, 1-\alpha}^2$ are flagged as anomalies.

---

### 5.2 Density-Based Anomaly Detection (Local Outlier Factor — LOF)

LOF measures local density deviation relative to a point's $k$-nearest neighbors.

```text
                       Local Outlier Factor (LOF) Topology
                     
              Dense Normal Cluster            Isolated Outlier (Point p)
                 ●   ●   ●                           ▲
               ●   ●   ●   ●                         │ High reach-dist,
                 ●   ●   ●                           │ low lrd density
                                                     │
                                                     ●

```

#### Mathematical Steps for LOF:

1. **$k$-Distance ($d_k(\mathbf{p})$):** The Euclidean distance $d(\mathbf{p}, \mathbf{o})$ between point $\mathbf{p}$ and its $k$-th nearest neighbor $\mathbf{o} \in \mathcal{D}$.
2. **$k$-Distance Neighborhood ($N_k(\mathbf{p})$):** The set of all points within distance $d_k(\mathbf{p})$.
3. **Reachability Distance:**

$$\text{reach-dist}_k(\mathbf{p}, \mathbf{o}) = \max \left\{ d_k(\mathbf{o}), \, d(\mathbf{p}, \mathbf{o}) \right\}$$

4. **Local Reachability Density ($\text{lrd}_k(\mathbf{p})$):** Inverse of the average reachability distance of point $\mathbf{p}$ from its neighbors:

$$\text{lrd}_k(\mathbf{p}) = \left[ \frac{\sum_{\mathbf{o} \in N_k(\mathbf{p})} \text{reach-dist}_k(\mathbf{p}, \mathbf{o})}{\vert{}N_k(\mathbf{p})\vert{}} \right]^{-1}$$

5. **Local Outlier Factor Score ($\text{LOF}_k(\mathbf{p})$):**

$$\text{LOF}_k(\mathbf{p}) = \frac{\sum_{\mathbf{o} \in N_k(\mathbf{p})} \frac{\text{lrd}_k(\mathbf{o})}{\text{lrd}_k(\mathbf{p})}}{\vert{}N_k(\mathbf{p})\vert{}}$$

* $\text{LOF} \approx 1$: Similar density to neighbors (Normal Point).
* $\text{LOF} < 1$: Higher density than neighbors (Inlier).
* $\text{LOF} \gg 1$: Significantly lower density than neighbors (Anomalous Point).

---

### 5.3 Isolation-Based Outlier Detection (Isolation Forest)

Isolation Forest explicitly isolates anomalies instead of profiling normal points. Because anomalies are rare and distinct, they require fewer random partition splits to isolate in an Isolation Tree (iTree).

```text
                   Isolation Tree Path Length Contrast
                   
       Normal Point Isolation (Deep Path)         Anomalous Point (Short Path)
                 [Root Node]                             [Root Node]
                /           \                           /           \
           [Split 1]      [Split 2]                (Anom)          [Split 1]
          /        \
     [Split 3]   [Split 4]
       /
  (Normal)

```

#### 1. Average Path Length of Unsuccessful Binary Search Tree Search:

Given a dataset of size $n$, the average path length of an unsuccessful search in a Binary Search Tree (BST) is:

$$c(n) = 2 \ln(n - 1) + 0.5772156649 \text{ (Euler's Constant)} - \frac{2(n - 1)}{n}$$

#### 2. Anomaly Score Formulation ($s(\mathbf{x}, n)$):

Given path length $h(\mathbf{x})$ across an ensemble of $T$ isolation trees:

$$s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$$

Where $\mathbb{E}(h(\mathbf{x}))$ is the average path length across all isolation trees.

* If $\mathbb{E}(h(\mathbf{x})) \to 0 \implies s \to 1$: The instance is **definitely anomalous** (isolated very close to the tree root).
* If $\mathbb{E}(h(\mathbf{x})) \to c(n) \implies s \to 0.5$: The instance exhibits **normal behavior**.
* If $\mathbb{E}(h(\mathbf{x})) \to n - 1 \implies s \to 0$: The instance is **deeply nested inside a dense cluster**.

---

### 5.4 One-Class Boundary Estimators (OC-SVM & SVDD)

One-Class Support Vector Machines (OC-SVM) project data into a high-dimensional feature space via a kernel function $\Phi(\mathbf{x})$ and separate normal instances from the origin using a maximum-margin hyperplane.

#### Support Vector Data Description (SVDD):

SVDD constructs a minimum-radius hyper-sphere surrounding normal data in feature space:

$$\min_{R, \mathbf{a}, \boldsymbol{\xi}} R^2 + C \sum_{i=1}^N \xi_i$$

$$\text{Subject to: } \Vert{}\Phi(\mathbf{x}_i) - \mathbf{a}\Vert{}^2 \le R^2 + \xi_i, \quad \xi_i \ge 0 \quad \forall i$$

Where $\mathbf{a}$ is the sphere center, $R$ is the radius, $\xi_i$ are slack variables, and $C$ controls the penalty for misclassifying normal points.

---

### 5.5 Deep Reconstruction-Based Methods (Autoencoders & VAEs)

Autoencoders leverage bottleneck network architectures to learn low-dimensional latent representations of normal data patterns.

```text
                Autoencoder Reconstruction Error Architecture
                
   Input x ∈ ℝ^d ──► [ Encoder ] ──► Latent Code z ∈ ℝ^k ──► [ Decoder ] ──► Output x̂ ∈ ℝ^d
                                                                               │
                                                                               ▼
                                                            Reconstruction Loss: ‖x - x̂‖²

```

#### 1. Optimization Objective (Training on Normal Instances):

$$\mathcal{L}_{\text{Reconstruction}}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \frac{1}{N} \sum_{i=1}^N \Vert{}\mathbf{x}_i - g_{\boldsymbol{\phi}}(f_{\boldsymbol{\theta}}(\mathbf{x}_i))\Vert{}_2^2$$

Where $f_{\boldsymbol{\theta}}$ is the encoder and $g_{\boldsymbol{\phi}}$ is the decoder.

#### 2. Anomaly Decision Rule:

Anomalous points $\mathbf{x}_{\text{anom}}$ contain novel patterns the autoencoder did not observe during training, yielding high reconstruction error:

$$\text{Score}(\mathbf{x}) = \Vert{}\mathbf{x} - \hat{\mathbf{x}}\Vert{}_2^2 \ge \tau \implies \text{Flagged as Anomaly}$$

---

## 6. Enterprise Latent Segmentation Architecture

In enterprise pipelines (such as fraud detection or cybersecurity), streaming events pass through feature processors, isolation estimators, reconstruction models, and thresholding modules.

```text
               Production Real-Time Anomaly Detection Engine
┌─────────────────────────────────────────────────────────────────────────────┐
│ HIGH-THROUGHPUT STREAMING EVENT INPUT (Kafka / Kinesis)                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE TRANSFORMATION & SCALING LAYER                                      │
│ • StandardScaler / RobustScaler & Feature Aggregations                      │
│ • Dimensionality Reduction (PCA / UMAP) for high-dimensional feature vectors │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DUAL DETECTOR INFERENCE ENGINE                                              │
│ ┌───────────────────────────────────┐     ┌───────────────────────────────┐ │
│ │ Model A: Isolation Forest          │     │ Model B: Autoencoder Network  │ │
│ │ (Fast partition path scoring)     │     │ (Reconstruction Error Loss)   │ │
│ └─────────────────┬─────────────────┘     └───────────────┬───────────────┘ │
└───────────────────┼───────────────────────────────────────┼─────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ENSEMBLE SCORING & DYNAMIC THRESHOLDING (EVT)                               │
│ • Extreme Value Theory (EVT) / Peak-Over-Threshold (POT) Decision Gate      │
│ • Score ≥ τ_dynamic ──► Flag Alert & Forward to Incident Queue             │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Metric & Selection Matrix

Standard accuracy is ineffective for evaluating anomaly detection because of severe class imbalance (often $< 1\%$ anomalies). Evaluation relies on ranking and threshold-independent metrics.

| Evaluation Metric | Target Type | Optimal Score | Key Characteristics & Best Use Cases |
| --- | --- | --- | --- |
| **ROC-AUC (Area Under ROC Curve)** | Probabilistic / Ranking | $1.0$ | Evaluates True Positive Rate vs. False Positive Rate across all thresholds. Can give an overly optimistic view under severe class imbalance. |
| **PR-AUC (Precision-Recall AUC)** | Probabilistic / Ranking | $1.0$ | Measures Precision vs. Recall. **Preferred metric for highly imbalanced anomaly detection datasets**. |
| **Precision@K ($P@K$)** | Top-$K$ Predictions | $1.0$ | Measures the proportion of true anomalies contained within the top-$K$ highest anomaly scores. Highly practical for manual triage. |
| **Recall@K ($R@K$)** | Top-$K$ Predictions | $1.0$ | Measures the percentage of total ground-truth anomalies captured within top-$K$ scoring instances. |
| **Adjusted F1 Score ($F_\beta$)** | Fixed Threshold $\tau$ | $1.0$ | Weighted harmonic mean of precision and recall. $F_2$ emphasizes recall for high-risk domains (e.g., medical diagnoses, fraud). |

---

## 8. Technology & Implementation Matrix

| Library / Tool | Core Algorithms Implemented | Primary Features | Target Production Environment |
| --- | --- | --- | --- |
| **PyOD** | `IForest`, `LOF`, `OCSVM`, `COPOD`, `ECOD`, `AutoEncoder` | Benchmark library containing over 40 anomaly detection algorithms with unified APIs. | Primary framework for statistical & unsupervised outlier benchmarking. |
| **Scikit-Learn** | `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`, `EllipticEnvelope` | Standard machine learning utilities, baseline anomaly estimators, and pre-processing modules. | General enterprise machine learning pipelines. |
| **PyTorch / TensorFlow** | Custom Autoencoders, VAEs, GANs (`AnoGAN`), LSTM Autoencoders | Deep reconstruction networks for unstructured text, images, and temporal sequences. | Complex deep learning and time-series anomaly systems. |
| **Alibi Detect** | Drift & Outlier Detectors (`SpectralResidual`, `Mahalanobis`) | Advanced model monitoring, outlier detection, and feature drift identification. | MLOps model monitoring and production data quality pipelines. |

---

## 9. Personal Understanding

Task 09 clarifies how anomaly detection algorithms identify rare, deviating observations without relying on balanced target labels.

Key personal insights include:

1. **Model assumptions must match the underlying geometry of anomalies:** Global distance methods (Z-score, Mahalanobis) fail on complex multi-modal distributions. Local density methods (LOF) handle variable-density clusters well, but scale poorly to large datasets ($\mathcal{O}(N^2)$). Isolation Forest scales efficiently ($\mathcal{O}(N \log N)$) and works well on high-dimensional data because it isolates sparse observations directly.
2. **Reconstruction models excel at complex patterns:** Autoencoders handle non-linear relationships by learning compressed latent spaces of normal behavior. If an input point cannot be reconstructed accurately, it deviates from the learned normal manifold, generating a high reconstruction error signal.
3. **Class imbalance makes metric selection critical:** Accuracy is ineffective when anomalies represent $< 0.1\%$ of the data (predicting all points as normal yields $99.9\%$ accuracy). Precision-Recall AUC (PR-AUC) and Precision@K are much more informative for evaluating model performance under extreme imbalance.

The core principle remains:

> **Anomaly detection algorithms isolate rare pattern occurrences by defining normal structural distributions, measuring local or global spatial density deviations, or computing reconstruction boundaries within feature representations.**

---

## 10. Interview / Viva Questions

### Q1. What is the difference between Point, Contextual, and Collective Anomalies?

**Answer:**

* **Point Anomaly:** An individual data point that lies far outside the distribution of the rest of the dataset (e.g., an individual transaction of $\$100,000$ on a credit card typically used for $\$20$ purchases).
* **Contextual Anomaly:** A data instance that is normal in general, but anomalous within a specific context (e.g., a temperature reading of $35^\circ\text{C}$ in winter vs. summer).
* **Collective Anomaly:** A collection of related data points that appear normal individually, but whose occurrence together as a sequence signals an anomaly (e.g., a sudden sequence of repeated small micro-transactions indicating a card-testing attack).

### Q2. Why is traditional classification accuracy a misleading metric for anomaly detection?

**Answer:**

Anomaly detection datasets suffer from severe class imbalance, with anomalies often making up less than $0.1\%$ of all observations. A trivial classifier that predicts every point as "Normal" achieves $99.9\%$ accuracy while missing every single anomaly. Therefore, metrics like **PR-AUC**, **ROC-AUC**, **Precision@K**, and **$F_\beta$-Score** are used instead.

### Q3. How does Isolation Forest isolate anomalies faster than normal data points?

**Answer:**

Isolation Forest constructs random decision trees by randomly selecting features and split values. Because anomalies reside in sparse regions of the feature space and have distinct values, they require fewer random splits to isolate from other points. As a result, anomalous points have significantly shorter path lengths from the tree root ($\mathbb{E}(h(\mathbf{x})) \ll c(n)$), yielding higher anomaly scores ($s \to 1.0$).

### Q4. Explain the mathematical intuition behind Local Outlier Factor (LOF).

**Answer:**

LOF compares the local reachability density ($\text{lrd}$) of a point $\mathbf{p}$ with the local densities of its $k$-nearest neighbors. If a point has a similar density to its neighbors, its LOF score is close to $1.0$. If a point lies in a sparse region while its neighbors lie in a dense cluster, its $\text{lrd}$ is much lower than its neighbors', resulting in an $\text{LOF} \gg 1.0$, which identifies it as a local outlier.

### Q5. What is the average path length $c(n)$ in Isolation Forest, and why is it used for normalization?

**Answer:**

$c(n) = 2 \ln(n - 1) + 0.5772156649 - \frac{2(n - 1)}{n}$ represents the average path length of an unsuccessful search in a binary search tree built from $n$ nodes. It serves as a normalization factor for the path length $h(\mathbf{x})$, ensuring the anomaly score $s(\mathbf{x}, n) = 2^{-\frac{\mathbb{E}(h(\mathbf{x}))}{c(n)}}$ produces bounded values between $0.0$ and $1.0$ regardless of sample size $n$.

### Q6. How does One-Class Support Vector Machine (OC-SVM) differ from standard binary SVM?

**Answer:**

Standard binary SVM finds a maximum-margin hyperplane that separates two distinct classes ($y \in \{-1, +1\}$). **One-Class SVM** is trained exclusively on normal data ($y = +1$). It maps data into a high-dimensional feature space using a kernel function $\Phi(\mathbf{x})$ and finds a hyperplane that separates the normal data points from the origin with maximum margin, treating the origin as the implicit surrogate for anomalies.

### Q7. Why is Mahalanobis distance superior to Euclidean distance for multivariate anomaly detection?

**Answer:**

Euclidean distance assumes features are uncorrelated and measured on identical scales, creating spherical distance contours. **Mahalanobis distance** accounts for variance scale and feature correlations using the covariance matrix $\mathbf{\Sigma}$:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

This forms elliptical decision boundaries that correctly identify points as outliers even if their individual feature values fall within normal univariate ranges.

### Q8. How does an Autoencoder detect anomalies in an unsupervised pipeline?

**Answer:**

An Autoencoder is trained to compress normal data instances into a lower-dimensional latent bottleneck and reconstruct them with minimal loss ($\Vert{}\mathbf{x} - \hat{\mathbf{x}}\Vert{}^2$). Because the network only learns patterns common to normal data, it struggles to reconstruct unseen anomalous patterns. When an anomaly is passed through the network, its reconstruction error is significantly higher than that of normal data, serving as an anomaly score.

### Q9. Why is PR-AUC preferred over ROC-AUC in severe class imbalance settings?

**Answer:**

ROC-AUC plots True Positive Rate ($\frac{TP}{TP + FN}$) against False Positive Rate ($\frac{FP}{FP + TN}$). In datasets with a large majority of true negatives ($TN$), the denominator ($FP + TN$) remains very large, keeping $FPR$ low even when the model makes many false positive predictions. PR-AUC plots Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{TP + FN}$), making it sensitive to false positives and giving a more accurate view of model performance under extreme class imbalance.

### Q10. What limitation of standard Isolation Forest does Extended Isolation Forest (EIF) resolve?

**Answer:**

Standard Isolation Forest selects axis-aligned hyperplanes (splitting parallel to feature axes), which creates artifact regions of low anomaly scores along axis directions (ghost anomalies). **Extended Isolation Forest (EIF)** uses hyperplanes with random slopes and random normal vectors ($\mathbf{n} \cdot (\mathbf{x} - \mathbf{p}) \le 0$), eliminating axis-aligned artifacts and improving detection accuracy on complex data manifolds.

### Q11. What is the contamination hyperparameter in anomaly detection libraries (e.g., PyOD, Scikit-Learn)?

**Answer:**

The `contamination` parameter specifies the expected proportion of outliers present in the dataset (e.g., `contamination=0.01` for $1\%$). It sets the decision threshold $\tau$ on calculated anomaly scores. Points with scores above $\tau$ are flagged as anomalies ($y = 1$), while points below are marked as normal ($y = 0$).

### Q12. How does high dimensionality affect distance and density-based anomaly detectors?

**Answer:**

High dimensionality causes spatial data sparsity, a key challenge associated with the **Curse of Dimensionality**. As dimensions increase, pairwise distances (e.g., Euclidean distance) converge to a uniform value, reducing the contrast between true outliers and normal points. Distance and density-based models (such as $k$-NN and LOF) become less effective unless paired with dimensionality reduction methods like PCA or UMAP.

### Q13. Compare Unsupervised, Semi-Supervised, and Supervised Anomaly Detection.

**Answer:**

* **Unsupervised:** No ground-truth labels are available; algorithms assume anomalies are rare points that differ structurally from the rest of the dataset (e.g., Isolation Forest, LOF).
* **Semi-Supervised (Novelty Detection):** The training set contains only clean, normal instances ($y = 0$). The model learns a baseline representation of normal behavior and flags deviations during inference (e.g., One-Class SVM, Autoencoders).
* **Supervised:** The dataset contains explicit ground-truth labels for both normal and anomalous classes ($y \in \{0, 1\}$). Standard imbalanced classification techniques are used (e.g., XGBoost with SMOTE, Focal Loss).

### Q14. What is dynamic thresholding, and how does Extreme Value Theory (EVT) apply to anomaly scoring?

**Answer:**

Fixed static thresholds ($\tau$) can lead to false alarms when data distributions drift over time. **Dynamic Thresholding** updates $\tau$ continuously using statistical sliding windows or **Extreme Value Theory (EVT)**. EVT uses the Peak-Over-Threshold (POT) approach, fitting a Generalized Pareto Distribution (GPD) to extreme tail values to calculate statistically grounded decision boundaries for rare events.

### Q15. Compare Variational Autoencoders (VAEs) and Standard Autoencoders for anomaly detection.

**Answer:**

Standard Autoencoders map inputs to fixed latent vectors, which can lead to irregular latent spaces with unconstrained gaps. **Variational Autoencoders (VAEs)** map inputs to mean $\boldsymbol{\mu}$ and variance $\boldsymbol{\sigma}^2$ vectors, enforcing a continuous Gaussian prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$ via Kullback-Leibler (KL) Divergence loss. This produces a smooth latent distribution, allowing VAEs to generate more robust reconstruction error signals and probabilistic anomaly scores.

---

## 11. Conclusion

Task 09 provides a thorough analysis of Anomaly Detection Formulations, Isolation Architectures, and Evaluation Metrics.

```text
                  Anomaly Detection Pipeline Flow
                                ↓
Dataset Ingestion & Feature Normalization (StandardScaler / RobustScaler)
                                ↓
Dimensionality Reduction / Manifold Mapping (PCA / UMAP for high-d vectors)
                                ↓
Detector Execution (Isolation Forest for scale, LOF for local density, AE for deep patterns)
                                ↓
Ensemble Scoring & Extreme Value Theory (EVT) Threshold Selection
                                ↓
Production Anomaly Alert Dispatch & Incident Auditing

```

The core structural pillars of Anomaly Detection include:

```text
Anomaly Detection Pillars
├── Parametric & Statistical Bounds (Z-Score, MAD, IQR, Mahalanobis Distance)
├── Local Density & Distance Reachability (k-NN distance, LOF local reachability)
├── Tree Partitioning & Isolation Mechanics (Isolation Forest, Extended IForest)
└── Boundary Support Vectors & Reconstruction (OC-SVM, SVDD, Autoencoders, VAEs)

```

Core tools and operational frameworks:

```text
PyOD Framework (Unified benchmarking for 40+ statistical & machine learning detectors)
Scikit-Learn (Production-ready IsolationForest, LOF, and OneClassSVM implementations)
PyTorch / TensorFlow (Deep Autoencoders, Variational Autoencoders, LSTM anomaly networks)
Alibi Detect (Model monitoring, drift identification, and online outlier detection)

```

Completing Task 09 provides the core understanding required to build production outlier detection systems, monitor streaming data pipelines for novel events, and evaluate models reliably under severe class imbalance.

The central principle remains:

> **Anomaly detection algorithms isolate rare pattern occurrences by defining normal structural distributions, measuring local or global spatial density deviations, or computing reconstruction boundaries within feature representations.**

---

## 12. Key Takeaways

1. **Anomaly Detection** identifies rare points, contextual events, or sequence patterns that deviate significantly from baseline normal behavior.
2. **Point Anomalies** are isolated individual points lying outside main feature distributions; **Contextual Anomalies** depend on contextual features like time or location.
3. **Collective Anomalies** consist of sequences of data points that are anomalous when observed together, even if individual points appear normal.
4. **Z-Score** assumes normal data distributions, whereas **Modified Z-Score (MAD)** provides robust outlier bounds using median absolute deviations.
5. **Mahalanobis Distance** incorporates feature covariance matrices, enabling multivariate outlier detection that accounts for feature scale and correlations.
6. **Local Outlier Factor (LOF)** calculates local reachability density ratios against $k$-nearest neighbors to identify density-based outliers.
7. **Isolation Forest** isolates anomalies directly using random split trees, relying on the principle that anomalies yield shorter average path lengths ($\mathbb{E}(h(\mathbf{x})) \ll c(n)$).
8. **Extended Isolation Forest (EIF)** uses hyperplanes with random slopes, eliminating axis-aligned artifact regions in high-dimensional spaces.
9. **One-Class SVM** maps data into a kernel space and finds a maximum-margin hyperplane separating normal points from the origin.
10. **Support Vector Data Description (SVDD)** computes a minimum-radius hyper-sphere surrounding normal instances in feature space.
11. **Autoencoder Detectors** train exclusively on normal data; instances with high reconstruction loss ($\|\mathbf{x} - \hat{\mathbf{x}}\|^2 \ge \tau$) are flagged as anomalies.
12. **Accuracy is misleading** for imbalanced anomaly detection; evaluation relies on **PR-AUC**, **ROC-AUC**, and **Precision@K**.
13. **PR-AUC** is preferred over ROC-AUC for severe class imbalance because it focuses on true positives relative to false positive spikes.
14. **Curse of Dimensionality** reduces contrast in distance metrics, requiring pre-processing via PCA, UMAP, or feature selection.
15. **Enterprise Anomaly Pipelines** combine ensemble scoring, streaming transformations, and dynamic thresholding (e.g., Extreme Value Theory) to prevent false alarm drift.
