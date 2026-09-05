# Task 01 — Mathematics for Data Science: Linear Algebra, Vector Spaces, Matrix Factorizations & Linear Transformations

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VI |
| Task Number | 01 (Foundational Task) |
| Topic | Mathematics for Data Science: Linear Algebra, Vector Spaces, Matrix Operations, Matrix Factorizations (LU, QR, SVD, Eigendecomposition), Norms, Rank, and Linear Transformations |
| Task Type | Applied Mathematics, Theoretical Linear Algebra & Core Data Science Foundations |
| Status | Completed |
| Repository Section | `tasks/portal-06/task-01/` |

---

## 2. Objective

The objective of this task is to formalize, analyze, and apply core concepts in **Linear Algebra, Matrix Decompositions, Vector Spaces, and Linear Transformations** as applied to modern Data Science and Machine Learning.
This task focuses on:
- Formalizing vector spaces, subspaces, span, linear independence, basis, and dimension.
- Analyzing matrix operations, rank, determinants, matrix inverses, and systems of linear equations ($A \mathbf{x} = \mathbf{b}$).
- Deriving vector norms ($L_1, L_2, L_\infty$) and matrix norms (Frobenius norm, Spectral norm) for regularization and error bounds.
- Mastering fundamental matrix factorizations: LU, QR, Cholesky, Eigendecomposition, and Singular Value Decomposition (SVD).
- Linking linear algebra directly to data science algorithms, including Principal Component Analysis (PCA), Ordinary Least Squares (OLS) regression, word embeddings, and low-rank matrix approximation.

---

## 3. Introduction

Linear algebra provides the fundamental language and computational engine for data science, machine learning, and deep learning. High-dimensional datasets are natively represented as matrices where rows correspond to samples and columns represent features. **Every linear model, neural network layer, dimensionality reduction technique, and optimization algorithm relies on transforming vector spaces via matrix operations.**

```text
               Linear Algebra Data Transformation Engine
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW HIGH-DIMENSIONAL DATASET (Matrix X ∈ ℝ^{m × n})                         │
│ (m samples, n feature dimensions; vector space basis ℝⁿ)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         [ Linear Transformation A ]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SUBSPACE PROJECTION & GEOMETRIC TRANSFORMATION                              │
│ Rotation, Scaling, Shearing ──► Subspace Range(A) & Nullspace Null(A)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                   [ Spectral / Low-Rank Factorization ]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPECTRUM & DIMENSION REDUCTION                                              │
│ Eigenvalues λ / Singular Values σ ──► PCA / Low-Rank Matrix Approximation   │
└─────────────────────────────────────────────────────────────────────────────┘

```

The core principle governing data science linear algebra is:

> **Linear algebra transforms high-dimensional data spaces into structured vector representations, enabling efficient transformations, projections, and low-rank approximations across data science algorithms.**

---

## 4. Paradigm Comparison Matrix

Comparing algebraic operations highlights essential trade-offs among numerical stability, computational complexity, memory overhead, and functional applicability in machine learning.

```text
            Linear Algebra Operations Comparison Matrix
┌─────────────────────────┬───────────────────────────────────────────────────┐
│ Algebraic Framework     │ Operational Execution & Mathematical Characteristics│
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Matrix Multiplication   │ Maps vectors across spaces $T: \mathbb{R}^n \to   │
│ ($A B$)                 │ \mathbb{R}^m$; $O(m n p)$ complexity; foundation  │
│                         │ of neural network forward passes and projections.  │
├─────────────────────────┼───────────────────────────────────────────────────┤
│ System Solver           │ Finds optimal solution $\mathbf{x}$; Normal Eqn  │
│ ($A \mathbf{x} = \mathbf{b}$)│ $\mathbf{x} = (A^T A)^{-1} A^T \mathbf{b}$; fragile │
│                         │ to ill-conditioned matrices ($\text{cond}(A) \gg 1$).│
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Eigendecomposition      │ Factorizes square symmetric matrix $A = Q \Lambda │
│ ($A = Q \Lambda Q^T$)   │ Q^T$; extracts variance axes; $O(n^3)$ complexity;│
│                         │ restricted to square, diagonalizable matrices.    │
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Singular Value          │ General factorization $A = U \Sigma V^T$ for any  │
│ Decomposition (SVD)     │ $m \times n$ matrix; optimal low-rank $k$-approx; │
│                         │ highly robust; $O(\min(m^2 n, m n^2))$ complexity.│
└─────────────────────────┴───────────────────────────────────────────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

Understanding linear algebra in data science requires formalizing vector spaces, matrix factorizations, vector/matrix norms, and projection mechanics.

### 5.1 Vector Spaces, Linear Independence, Span, & Rank

A **Vector Space** $V$ over $\mathbb{R}$ is a set closed under vector addition and scalar multiplication satisfying associative, commutative, and distributive axioms.

#### Linear Independence & Span:

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is **linearly independent** if:

$$\sum_{i=1}^k c_i \mathbf{v}_i = \mathbf{0} \implies c_1 = c_2 = \dots = c_k = 0$$

If non-zero scalars exist, the vectors are linearly dependent. The **Span** is the set of all possible linear combinations of the vectors.

#### Matrix Rank & Rank-Nullity Theorem:

For a matrix $A \in \mathbb{R}^{m \times n}$:

* **Column Rank:** Maximum number of linearly independent column vectors.
* **Row Rank:** Maximum number of linearly independent row vectors. $\text{rank}(A) \le \min(m, n)$.
* **Rank-Nullity Theorem:**

$$\text{rank}(A) + \text{nullity}(A) = n$$

Where $\text{nullity}(A) = \text{dim}(\text{Null}(A))$, representing the dimension of solutions to $A \mathbf{x} = \mathbf{0}$.

---

### 5.2 Vector Norms, Matrix Norms, & System Solvers

Norms quantify size, distance, and magnitude, serving as the mathematical backbone of loss functions and regularization penalties ($L_1$ Lasso, $L_2$ Ridge).

#### Vector $p$-Norm Definition:

$$\Vert{}\mathbf{x}\Vert{}_p = \left( \sum_{i=1}^n \vert{}x_i\vert{}^p \right)^{1/p}$$

1. **$L_1$ Norm (Manhattan):** $\Vert{}\mathbf{x}\Vert{}_1 = \sum \vert{}x_i\vert{}$ (Promotes sparsity in feature selection).
2. **$L_2$ Norm (Euclidean):** $\Vert{}\mathbf{x}\Vert{}_2 = \sqrt{\sum x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}$ (Measures geometric length).
3. **$L_\infty$ Norm (Max Norm):** $\Vert{}\mathbf{x}\Vert{}_\infty = \max_i \vert{}x_i\vert{}$.

#### Matrix Norms:

* **Frobenius Norm:** $\Vert{}A\Vert{}_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\text{Tr}(A^T A)}$
* **Spectral Norm ($L_2$ Matrix Norm):** $\Vert{}A\Vert{}_2 = \sigma_{\max}(A)$ (Largest singular value).

#### Ordinary Least Squares (OLS) Normal Equations:

For an overdetermined system $X \boldsymbol{\beta} \approx \mathbf{y}$ where $X \in \mathbb{R}^{m \times n}$ ($m > n$):

$$\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$$

---

### 5.3 Eigenvalues, Eigenvectors, & Eigendecomposition

For a square matrix $A \in \mathbb{R}^{n \times n}$, an **eigenvector** $\mathbf{v} \neq \mathbf{0}$ and **eigenvalue** $\lambda \in \mathbb{R}$ satisfy:

$$A \mathbf{v} = \lambda \mathbf{v} \implies (A - \lambda I) \mathbf{v} = \mathbf{0}$$

Nontrivial solutions exist if and only if the characteristic determinant equals zero:

$$\det(A - \lambda I) = 0$$

#### Spectral Theorem for Real Symmetric Matrices:

If $A = A^T$, then $A$ can be factorized into orthogonal eigenvectors $Q$ and diagonal matrix of eigenvalues $\Lambda$:

$$A = Q \Lambda Q^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T, \quad \text{where } Q^T Q = I$$

```text
                  Eigendecomposition Geometric Action
 [ Input Vector v ]  ──►  Matrix Scaling Matrix A  ──►  Stretched Vector λv
  (Direction Preserved)        (A v = λ v)            (Scaled by Eigenvalue λ)

```

---

### 5.4 Singular Value Decomposition (SVD)

SVD generalizes eigendecomposition to any arbitrary matrix $A \in \mathbb{R}^{m \times n}$ of rank $r$:

$$A = U \Sigma V^T$$

Where:

* $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix of **left singular vectors** (eigenvectors of $A A^T$).
* $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix of non-negative **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ (where $\sigma_i = \sqrt{\lambda_i(A^T A)}$).
* $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix of **right singular vectors** (eigenvectors of $A^T A$).

#### Eckart-Young-Mirsky Theorem (Truncated Low-Rank SVD Approximation):

The optimal rank-$k$ approximation ($k < r$) to matrix $A$ under the Frobenius norm is given by truncating SVD to the top $k$ singular values:

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T, \quad \min_{\text{rank}(B)=k} \Vert{}A - B\Vert{}_F = \Vert{}A - A_k\Vert{}_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

---

## 6. Enterprise Data Science Architecture

In modern production machine learning pipelines, high-dimensional vectorized features are processed using accelerated linear algebra libraries and GPU compute kernels.

```text
             Vectorized Data Processing Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ HIGH-DIMENSIONAL FEATURE STORE (Feature Matrix X ∈ ℝ^{N × D})               │
│ N = 10⁷ instances, D = 10³ dense/sparse numerical vectors                   │
└──────────────────────┬──────────────────────────────────────┬───────────────┘
                       │                                      │
                       ▼                                      ▼
┌────────────────────────────────────────┐  ┌─────────────────────────────────┐
│ GPU ACCELERATED LINEAR ALGEBRA ENGINE  │  │ DIMENSIONALITY REDUCTION (SVD)  │
│ CUDA / BLAS Tensor Multiplications     │  │ Computes Singular Values σ      │
│ Vectorized Batch Operations: W X + b   │  │ Projects X ──► Reduced Space Z  │
└──────────────────────┬─────────────────┘  └─────────────────┬───────────────┘
                       │                                      │
                       ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SUBSPACE & MANIFOLD ESTIMATION ENGINE                                       │
│ Covariance Estimation Σ = 1/N X^T X  ──►  Eigen-spectrum Analysis           │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DOWNSTREAM ML ALGORITHM EMBEDDINGS                                          │
│ Projected Feature Matrices ──► Ridge / Lasso Regression / Deep Learning     │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Analysis & Factorization Selection Matrix

Choosing the correct matrix decomposition technique depends on matrix properties (square, symmetric, rectangular), computational complexity, and task requirements.

| Factorization Technique | Input Matrix Conditions | Factorized Form | Primary Data Science Applications | Computational Complexity |
| --- | --- | --- | --- | --- |
| **LU Decomposition** | Square matrix $A \in \mathbb{R}^{n \times n}$. | $P A = L U$ ($L$: lower, $U$: upper triangular). | Efficient linear system solving ($A \mathbf{x} = \mathbf{b}$); matrix inversion. | $O\left(\frac{2}{3} n^3\right)$ |
| **QR Decomposition** | Rectangular matrix $A \in \mathbb{R}^{m \times n}$ ($m \ge n$). | $A = Q R$ ($Q$: orthogonal, $R$: upper triangular). | Numerically stable OLS regression; Gram-Schmidt orthogonalization. | $O(2 m n^2)$ |
| **Cholesky Decomposition** | Symmetric Positive-Definite $A = A^T, \mathbf{x}^T A \mathbf{x} > 0$. | $A = L L^T$ ($L$: lower triangular). | Fast Gaussian Process sampling; multivariate normal distribution simulation. | $O\left(\frac{1}{3} n^3\right)$ |
| **Eigendecomposition** | Square symmetric $A \in \mathbb{R}^{n \times n}$. | $A = Q \Lambda Q^T$ ($\Lambda$: eigenvalues diagonal). | Principal Component Analysis (PCA) on Covariance matrix $X^T X$; Markov chains. | $O(n^3)$ |
| **Singular Value Decomposition (SVD)** | Any rectangular $A \in \mathbb{R}^{m \times n}$. | $A = U \Sigma V^T$ ($U, V$: orthogonal, $\Sigma$: singular). | Latent Semantic Analysis (LSA); Recommender Systems; Low-Rank Compression. | $O(\min(m^2 n, m n^2))$ |
| **Moore-Penrose Pseudoinverse** | Any rectangular $A \in \mathbb{R}^{m \times n}$. | $A^+ = V \Sigma^+ U^T$. | Minimum-norm least-squares solving for singular/underdetermined systems. | $O(\min(m^2 n, m n^2))$ |

---

## 8. Technology & Integration Matrix

| Functional Role | Industry Standard Tooling | Primary Operational Function |
| --- | --- | --- |
| **Scientific CPU Algebra** | NumPy (`numpy.linalg`), SciPy (`scipy.linalg`) | High-performance CPU-bound vector, matrix operations, factorizations, and linear solvers using LAPACK/BLAS. |
| **GPU Tensor Multiplications** | PyTorch (`torch.linalg`), TensorFlow (`tf.linalg`) | Executes massive GPU-accelerated tensor arithmetic, automatic differentiation, and matrix factorizations. |
| **GPU Array Operations** | CuPy (`cupy.linalg`) | CUDA-backed drop-in replacement for NumPy for multi-gigabyte vector operations. |
| **Dimensionality Reduction** | Scikit-Learn (`sklearn.decomposition`) | Implements PCA, TruncatedSVD, FastICA, and NMF algorithms for feature selection and extraction. |
| **Sparse Matrix Algebra** | SciPy Sparse (`scipy.sparse.linalg`) | Efficient storage and spectral solvers for sparse adjacency matrices in Graph Neural Networks. |

---

## 9. Personal Understanding

Task 01 establishes the essential foundation for mathematical operations across data science and machine learning.
I now realize that **matrices are not merely static tables of numbers; they represent dynamic geometric transformations** that rotate, scale, stretch, and project vector spaces.
Machine learning models manipulate these geometric spaces to discover hidden structures. For instance, when fitting a linear regression model, we project target vector $\mathbf{y}$ onto the column space of feature matrix $X$. In Principal Component Analysis (PCA), we compute the eigenvectors of covariance matrix $X^T X$ to identify directions that maximize variance. SVD allows us to compress massive datasets into low-rank representations without losing essential structural information.
The central principle remains:

> **Linear algebra transforms high-dimensional data spaces into structured vector representations, enabling efficient transformations, projections, and low-rank approximations across data science algorithms.**

---

## 10. Interview / Viva Questions

### Q1. What is the difference between linear independence and linear dependence in a set of vectors?

**Answer:**

A set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is **linearly independent** if no vector in the set can be written as a linear combination of the others. Mathematically, $\sum_{i=1}^k c_i \mathbf{v}_i = \mathbf{0}$ holds if and only if all coefficients $c_i = 0$. If at least one non-zero coefficient $c_i \neq 0$ exists, the vectors are **linearly dependent**, meaning at least one vector lies within the span of the remaining vectors, causing dimension redundancy.

### Q2. Define the Rank of a matrix and explain the Rank-Nullity Theorem.

**Answer:**

The **Rank** of a matrix $A \in \mathbb{R}^{m \times n}$, denoted $\text{rank}(A)$, is the maximum number of linearly independent column (or row) vectors. It represents the dimension of the matrix's column space $\text{Col}(A)$.

The **Rank-Nullity Theorem** states that for an $m \times n$ matrix $A$:

$$\text{rank}(A) + \text{nullity}(A) = n$$

Where $n$ is the total number of columns, and $\text{nullity}(A) = \text{dim}(\text{Null}(A))$ is the dimension of the kernel/nullspace (the space of solutions to $A \mathbf{x} = \mathbf{0}$).

### Q3. What is the geometric interpretation of Eigenvalues and Eigenvectors?

**Answer:**

When a matrix $A$ acts as a linear transformation on a vector $\mathbf{x}$ ($A \mathbf{x}$), it generally alters both the direction and magnitude of $\mathbf{x}$. An **eigenvector** $\mathbf{v}$ is a special vector whose direction remains unchanged by transformation $A$; it is merely scaled by a scalar factor $\lambda$, known as the **eigenvalue** ($A \mathbf{v} = \lambda \mathbf{v}$). If $\lambda > 1$, space is stretched along $\mathbf{v}$; if $0 < \lambda < 1$, space is compressed; if $\lambda < 0$, direction is flipped.

### Q4. What is the difference between Eigendecomposition and Singular Value Decomposition (SVD)?

**Answer:**

* **Eigendecomposition ($A = Q \Lambda Q^T$):** Applicable **only to square matrices** $n \times n$. Uses a single basis of eigenvectors $Q$. Eigenvalues $\lambda$ can be negative or complex (unless $A$ is symmetric positive-definite).
* **Singular Value Decomposition ($A = U \Sigma V^T$):** Applicable to **any $m \times n$ rectangular matrix**. Uses two distinct orthogonal bases: left singular vectors $U$ and right singular vectors $V$. Singular values $\sigma_i$ are always real and non-negative ($\sigma_i \ge 0$).

### Q5. Why do we compute $(X^T X)^{-1} X^T \mathbf{y}$ in Ordinary Least Squares (OLS) regression?

**Answer:**

In linear regression $X \boldsymbol{\beta} = \mathbf{y}$, the target vector $\mathbf{y}$ typically does not lie within the column space $\text{Col}(X)$ due to noise. We seek a solution $\hat{\mathbf{y}} = X \hat{\boldsymbol{\beta}}$ in $\text{Col}(X)$ that minimizes the Euclidean distance $\Vert{}\mathbf{y} - X \boldsymbol{\beta}\Vert{}_2^2$. Geometrically, the shortest distance occurs when residual error vector $(\mathbf{y} - X \hat{\boldsymbol{\beta}})$ is orthogonal to $\text{Col}(X)$:

$$X^T (\mathbf{y} - X \hat{\boldsymbol{\beta}}) = \mathbf{0} \implies X^T X \hat{\boldsymbol{\beta}} = X^T \mathbf{y} \implies \hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$$

These are the **Normal Equations**.

### Q6. What is a Symmetric Positive-Definite (SPD) matrix, and why is it important in Data Science?

**Answer:**

A real symmetric matrix $A \in \mathbb{R}^{n \times n}$ ($A = A^T$) is **Positive-Definite** if for all non-zero vectors $\mathbf{x} \neq \mathbf{0}$:

$$\mathbf{x}^T A \mathbf{x} > 0$$

* **Key Properties:** All eigenvalues of an SPD matrix are strictly positive ($\lambda_i > 0$). Determinant is positive ($\det(A) > 0$).
* **Importance:** Covariance matrices $X^T X$ are always symmetric positive semi-definite. SPD matrices ensure convex optimization landscapes with unique global minima (e.g., loss function Hessian matrices).

### Q7. What are Orthogonal Matrices and what are their mathematical properties?

**Answer:**

A square matrix $Q \in \mathbb{R}^{n \times n}$ is **orthogonal** if its columns form an orthonormal basis:

$$Q^T Q = Q Q^T = I \implies Q^{-1} = Q^T$$

* **Properties:**
1. Preserves vector norms (length): $\Vert{}Q \mathbf{x}\Vert{}_2 = \Vert{}\mathbf{x}\Vert{}_2$.
2. Preserves dot products and angles: $(Q \mathbf{x})^T (Q \mathbf{y}) = \mathbf{x}^T \mathbf{y}$.
3. Determinant $\det(Q) = \pm 1$ (representing pure rotations or reflections).
4. Highly numerically stable in computational linear algebra.



### Q8. What is the difference between $L_1$ norm and $L_2$ norm, and how do they affect model regularization?

**Answer:**

* **$L_1$ Norm ($\Vert{}\mathbf{w}\Vert{}_1 = \sum \vert{}w_i\vert{}$):** Measures Manhattan distance. In regularized optimization (Lasso), $L_1$ penalizes absolute parameter values, driving small weights strictly to zero. This produces **sparse models** and performs automated feature selection.
* **$L_2$ Norm ($\Vert{}\mathbf{w}\Vert{}_2 = \sqrt{\sum w_i^2}$):** Measures Euclidean length. In Ridge regression, $L_2$ penalizes squared parameter values, shrinking weights toward zero without forcing them exactly to zero. This handles multicollinearity smoothly.

### Q9. What is the Condition Number of a matrix, and what does an ill-conditioned matrix mean?

**Answer:**

The **Condition Number** of a matrix $A$ under $L_2$ norm is the ratio of its maximum to minimum singular value:

$$\text{cond}(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

* **Interpretation:** It measures system sensitivity $A \mathbf{x} = \mathbf{b}$ to small perturbations in input $\mathbf{b}$.
* **Ill-Conditioned System:** If $\text{cond}(A) \gg 1$, $A$ is nearly singular (multicollinear features in regression). Small noise in data causes huge swings in solution $\mathbf{x}$, making matrix inversion $(X^T X)^{-1}$ numerically unstable.

### Q10. Explain the Gram-Schmidt process.

**Answer:**

The Gram-Schmidt process takes a set of linearly independent vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ and converts them into an **orthonormal** set $\{\mathbf{q}_1, \dots, \mathbf{q}_k\}$ spanning the exact same subspace:

1. First vector: $\mathbf{u}_1 = \mathbf{v}_1, \quad \mathbf{q}_1 = \frac{\mathbf{u}_1}{\Vert{}\mathbf{u}_1\Vert{}}$
2. Subsequent vectors subtract orthogonal projections onto previously computed bases:

$$\mathbf{u}_i = \mathbf{v}_i - \sum_{j=1}^{i-1} (\mathbf{v}_i^T \mathbf{q}_j) \mathbf{q}_j, \quad \mathbf{q}_i = \frac{\mathbf{u}_i}{\Vert{}\mathbf{u}_i\Vert{}}$$

This process forms the foundation of QR decomposition ($A = QR$).

### Q11. How is Principal Component Analysis (PCA) derived using Linear Algebra?

**Answer:**

For zero-mean data matrix $X \in \mathbb{R}^{N \times D}$, the covariance matrix is $S = \frac{1}{N} X^T X \in \mathbb{R}^{D \times D}$.
We seek a unit projection vector $\mathbf{w}$ ($\Vert{}\mathbf{w}\Vert{}_2 = 1$) that maximizes variance of projected data $X \mathbf{w}$:

$$\max_{\mathbf{w}} \text{Var}(X \mathbf{w}) = \max_{\mathbf{w}} \mathbf{w}^T S \mathbf{w} \quad \text{subject to } \mathbf{w}^T \mathbf{w} = 1$$

Using Lagrange multipliers $\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T S \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1)$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2 S \mathbf{w} - 2 \lambda \mathbf{w} = \mathbf{0} \implies S \mathbf{w} = \lambda \mathbf{w}$$

Thus, principal component directions $\mathbf{w}$ are the **eigenvectors** of covariance matrix $S$, and projected variance equals eigenvalue $\lambda$.

### Q12. What are Trace and Determinant, and how do they relate to matrix Eigenvalues?

**Answer:**

For a square matrix $A \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda_1, \dots, \lambda_n$:

* **Trace ($\text{Tr}(A)$):** Sum of main diagonal elements $\sum_{i=1}^n a_{ii}$. It equals the sum of eigenvalues: $\text{Tr}(A) = \sum_{i=1}^n \lambda_i$.
* **Determinant ($\det(A)$):** Represents volume scaling factor of transformation $A$. It equals the product of eigenvalues: $\det(A) = \prod_{i=1}^n \lambda_i$. If $\det(A) = 0$, at least one eigenvalue is 0, making $A$ singular and non-invertible.

### Q13. What is the Moore-Penrose Pseudoinverse ($A^+$) and when is it used?

**Answer:**

The **Moore-Penrose Pseudoinverse** $A^+$ generalizes matrix inversion to rectangular or singular matrices $A \in \mathbb{R}^{m \times n}$. Computed via SVD ($A = U \Sigma V^T$):

$$A^+ = V \Sigma^+ U^T$$

Where $\Sigma^+$ transposes $\Sigma$ and replaces all non-zero singular values $\sigma_i$ with their reciprocal $\frac{1}{\sigma_i}$.

* **Usage:** Solves least-squares systems $A \mathbf{x} = \mathbf{b}$ when $A$ is underdetermined (infinitely many solutions) or overdetermined with collinear columns. Provides unique minimum $L_2$-norm solution $\hat{\mathbf{x}} = A^+ \mathbf{b}$.

### Q14. What is a Projection Matrix and what is its fundamental property?

**Answer:**

A **Projection Matrix** $P$ projects a vector $\mathbf{x}$ orthogonally onto a subspace $S$.

* **Formula:** For a subspace spanned by columns of matrix $A$: $P = A (A^T A)^{-1} A^T$.
* **Properties:**
1. **Idempotent:** $P^2 = P$ (Projecting an already projected vector changes nothing).
2. **Symmetric:** $P^T = P$.
3. Complementary projection matrix $(I - P)$ projects onto the orthogonal complement $S^\perp$.



### Q15. Explain the Eckart-Young-Mirsky Theorem for low-rank matrix approximation.

**Answer:**

The Eckart-Young-Mirsky theorem states that given a matrix $A$ of rank $r$, the best rank-$k$ approximation ($k < r$) that minimizes reconstruction error under Frobenius norm (or Spectral norm) is obtained by taking the truncated SVD ($A_k = U_k \Sigma_k V_k^T$):

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

Reconstruction loss equals: $\Vert{}A - A_k\Vert{}_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$. This provides the theoretical foundation for image compression, SVD-based recommender systems, and TruncatedSVD in NLP.

---

## 11. Conclusion

Task 01 provides the mathematical foundation required to analyze high-dimensional vector spaces, implement linear models, execute dimensionality reduction, and build scalable machine learning architectures.
The complete linear algebra pipeline flow is summarized below:

```text
Linear Algebra Operations Lifecycle Flow
      ↓
High-Dimensional Vector Space Formulation (ℝⁿ Matrix Representation)
      ↓
Vector/Matrix Norm Quantification & Condition Number Evaluation (cond(A))
      ↓
Matrix Factorization Execution (LU / QR / Cholesky / Eigendecomposition / SVD)
      ↓
Subspace Projection & Dimensionality Reduction (PCA / Low-Rank Approximation)
      ↓
Downstream Vectorized Model Operations (Least Squares, Neural Passes)

```

The core structural pillars of Data Science Linear Algebra include:

```text
Data Science Linear Algebra Foundations
├── Vector Space Mechanics (Independence, Basis, Span, Rank-Nullity Theorem)
├── Norms & Solvers (L1/L2 Norms, Frobenius, OLS Normal Equations, Pseudoinverse)
├── Spectral Theory (Eigenvalues, Eigenvectors, Spectral Theorem, Positive Definite)
└── Matrix Factorizations & SVD (LU, QR, SVD, Low-Rank Approximation, PCA)

```

Core tools and operational frameworks:

```text
NumPy (numpy.linalg) / SciPy (scipy.linalg)
PyTorch (torch.linalg) / TensorFlow (tf.linalg) / CuPy
Scikit-Learn (PCA, TruncatedSVD)

```

By completing Task 01, data scientists master the core mathematical concepts, matrix operations, factorizations, and transformations needed to build reliable, computationally efficient machine learning models.
The central principle remains:

> **Linear algebra transforms high-dimensional data spaces into structured vector representations, enabling efficient transformations, projections, and low-rank approximations across data science algorithms.**

---

## 12. Key Takeaways

1. **Linear algebra** forms the core computational framework for vector spaces, dataset matrices, and neural networks.
2. **Matrices** act as linear transformations that stretch, rotate, scale, and project vector spaces.
3. A set of vectors is **linearly independent** if no vector can be expressed as a linear combination of others in the set.
4. **Matrix Rank** specifies the dimension of a matrix's column space; the **Rank-Nullity Theorem** links rank and nullity to total columns ($r + \text{nullity} = n$).
5. **$L_1$ Norm** enforces sparsity in model weights (Lasso), while **$L_2$ Norm** penalizes large weight magnitudes smoothly (Ridge).
6. **Ordinary Least Squares (OLS)** computes parameter weights via normal equations $\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}$ by projecting $\mathbf{y}$ onto $\text{Col}(X)$.
7. An **ill-conditioned matrix** ($\text{cond}(A) \gg 1$) exhibits strong multicollinearity, causing numerical instability during matrix inversion.
8. **Eigenvectors** maintain their spatial direction under linear transformation $A$, scaling strictly by **eigenvalues** $\lambda$ ($A \mathbf{v} = \lambda \mathbf{v}$).
9. Real symmetric matrices ($A = A^T$) can be orthogonally factorized into $A = Q \Lambda Q^T$ via the **Spectral Theorem**.
10. **Singular Value Decomposition (SVD)** factorizes *any* $m \times n$ matrix into $U \Sigma V^T$, extracting singular values $\sigma_i = \sqrt{\lambda_i(A^T A)}$.
11. The **Eckart-Young-Mirsky Theorem** guarantees that truncating SVD to $k$ singular values yields the optimal rank-$k$ matrix approximation under Frobenius norm.
12. **Principal Component Analysis (PCA)** uses eigenvectors of covariance matrix $X^T X$ to identify directions that maximize variance.
13. **Orthogonal matrices** ($Q^T Q = I$) preserve vector lengths and angles, providing numerical stability in linear computations.
14. The **Moore-Penrose Pseudoinverse ($A^+$)** computes optimal least-squares solutions for singular or rectangular matrices.
15. **Projection matrices** $P = A(A^T A)^{-1}A^T$ are idempotent ($P^2 = P$) and symmetric ($P^T = P$).
