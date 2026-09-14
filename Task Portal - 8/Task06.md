# Task 06 — Generative Adversarial Networks (GANs), Minimax Dynamics & Architectural Variants

## 1. Task Information

| Field | Details |
| --- | --- |
| Internship | Data Science Internship — Portal VIII |
| Task Number | Task 06 (Advanced Generative Modeling & Deep Learning) |
| Topic | Generative Adversarial Networks (GANs): Minimax Two-Player Zero-Sum Game Dynamics, Generator & Discriminator Loss Formulations, Vanishing Gradients & Non-Saturating Loss, Wasserstein GAN (WGAN & WGAN-GP), DCGAN Guidelines, Conditional GAN (cGAN), Cycle-Consistent GAN (CycleGAN), Mode Collapse Diagnostics, and Image Quality Metrics (IS & FID) |
| Task Type | Algorithmic Derivation, Minimax Equilibrium Proof, & Generative Architecture Design |
| Status | Completed |
| Repository Section | `tasks/portal-08/task-06/` |

---

## 2. Objective

The objective of this task is to provide an exhaustive mathematical, architectural, and empirical analysis of **Generative Adversarial Networks (GANs)** and their advanced variants.
This task covers:

* Formulating **Goodfellow's Minimax Two-Player Game**, deriving the theoretical optimal discriminator $D^*(x)$, and proving that global minimum convergence yields the Jensen-Shannon Divergence $\text{JSD}(p_{\text{data}} \parallel p_g)$.
* Analyzing **Training Pathologies**: vanishing gradients under saturated loss, mode collapse dynamics, non-convergence (limit cycles), and discriminator over-powering.
* Mathematical derivation of **Wasserstein GAN (WGAN)** using the Earth Mover's Distance, Kantorovich-Rubinstein Duality, 1-Lipschitz continuity constraints, Weight Clipping, and Gradient Penalty (WGAN-GP).
* Architectural engineering of **Deep Convolutional GAN (DCGAN)**, **Conditional GAN (cGAN)**, and **Cycle-Consistent GAN (CycleGAN)** with unpaired image-to-image translation.
* Diagnostic evaluation of synthetic image quality using **Inception Score (IS)** and **Fréchet Inception Distance (FID)**.

---

## 3. Introduction & Conceptual Framework

Introduced by Ian Goodfellow et al. (2014), **Generative Adversarial Networks (GANs)** frame generative modeling as a game-theoretic competition between two deep neural networks:

1. **The Generator ($G$):** Maps a latent random noise vector $z \sim p_z(z)$ to synthetic data instances $G(z)$ attempting to replicate the true data distribution $p_{\text{data}}$.
2. **The Discriminator ($D$):** A binary classifier outputting probability $D(x) \in [0, 1]$ that sample $x$ originates from real data distribution $p_{\text{data}}$ rather than synthetic distribution $p_g$.

---

```text
                     Generative Adversarial Feedback Loop
┌─────────────────────────┐
│ Latent Vector z ~ p_z   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐           Generated Fake x_fake
│ Generator Network G(z)  ├─────────────────────────────────────────┐
└─────────────────────────┘                                         │
                                                                    ▼
┌─────────────────────────┐ Real x_real                    ┌────────────────┐
│ Real Dataset p_data(x)  ├───────────────────────────────►│ Discriminator  │──► Prediction D(x)
└─────────────────────────┘                                │ Network D      │    (1=Real, 0=Fake)
                                                           └───────┬────────┘
                                                                   │
                                  ┌────────────────────────────────┴────────────────┐
                                  ▼                                                 ▼
                     Discriminator Loss L_D                            Generator Loss L_G
                     (Maximize Log D(x) + Log(1-D(G(z))))             (Minimize Log(1-D(G(z))))
                                  │                                                 │
                                  └───────────────► Backpropagation ◄───────────────┘

```

The fundamental governing axiom of adversarial generative modeling is:

> **Generative Adversarial Networks frame density estimation as a two-player zero-sum game where non-cooperative minimax dynamics converge to a Nash Equilibrium when the implicit generated probability distribution $p_g$ matches the empirical data distribution $p_{\text{data}}$.**

---

## 4. Architectural Variants Comparison Matrix

| GAN Variant | Loss Formulation Paradigm | Continuity / Stability Mechanism | Primary Target Capability | Key Operational Vulnerability |
| --- | --- | --- | --- | --- |
| **Vanilla GAN** | Binary Cross-Entropy Minimax Game | Non-saturating heuristic ($\max \log D(G(z))$) | Unconditional tabular / image synthesis | Extreme mode collapse & vanishing gradients |
| **DCGAN** | Strided Convolutional Cross-Entropy | Batch Normalization & LeakyReLU ($\alpha=0.2$) | Stable spatial image generation | Sensitive to hyperparameter learning rates |
| **WGAN** | Earth Mover's Distance (Wasserstein-1) | Weight Clipping $[-c, c]$ (1-Lipschitz) | Continuous gradient signal across splits | Capacity degradation & vanishing/exploding weights |
| **WGAN-GP** | Earth Mover's Distance + Gradient Penalty | Soft 1-Lipschitz constraint $(\Vert{}\nabla_{\hat{x}} D(\hat{x})\Vert{}_2 - 1)^2$ | High-resolution stable generative modeling | $\sim 3\times$ higher computational training overhead |
| **cGAN** | Label-Conditioned Minimax Objective | Concatenation of target label vector $y$ | Class-steered generation (e.g., specific digit) | Requires complete, clean label annotations |
| **CycleGAN** | Dual Adversarial + Cycle Consistency | $L_1$ Cycle Loss $\Vert{}F(G(X)) - X\Vert{}_1$ | Unpaired domain-to-domain image translation | Fails on severe geometric shape transformations |

---

## 5. Mathematical & Algorithmic Foundations

---

### 5.1 Goodfellow's Two-Player Minimax Formulation

The training of Vanilla GANs is formulated as a zero-sum game with value function $V(D, G)$:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]$$

Rewriting the expected value over the latent distribution as an integral over generated distribution $p_g$:

$$V(D, G) = \int_{\mathcal{X}} p_{\text{data}}(x) \log D(x) \, dx + \int_{\mathcal{X}} p_g(x) \log (1 - D(x)) \, dx$$

#### Derivation of Optimal Discriminator $D^*_G(x)$:

To find the optimal discriminator for any fixed generator $G$, maximize the integrand function $f(y) = a \log(y) + b \log(1 - y)$ where $a = p_{\text{data}}(x)$, $b = p_g(x)$, and $y = D(x)$:

$$\frac{df(y)}{dy} = \frac{a}{y} - \frac{b}{1 - y} = 0 \implies a(1 - y) - by = 0 \implies y^* = \frac{a}{a + b}$$

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

#### Global Optimum Proof ($p_g = p_{\text{data}}$):

Substituting $D^*_G(x)$ back into the objective function $V(D^*_G, G)$:

$$C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}\right] + \mathbb{E}_{x \sim p_g}\left[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}\right]$$

Adding and subtracting $\log 2$ terms converts this expression into Kullback-Leibler (KL) divergences:

$$C(G) = -\log 4 + \text{KL}\left(p_{\text{data}} \,||\, \frac{p_{\text{data}} + p_g}{2}\right) + \text{KL}\left(p_g \,||\, \frac{p_{\text{data}} + p_g}{2}\right)$$

By definition of **Jensen-Shannon Divergence (JSD)**:

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)$$

Since $\text{JSD}(p_{\text{data}} \parallel p_g) \ge 0$ and equals zero if and only if $p_g = p_{\text{data}}$, the absolute minimum of the generator loss is $C^* = -\log 4$, achieved uniquely when the generated distribution perfectly matches the real data distribution.

---

### 5.2 Training Pathologies & Gradient Dynamics

```text
                     GAN Training Pathologies Framework
                     
  Saturating Minimax Loss ──► Discriminator wins early ──► Vanishing gradient to Generator
  Disjoint Support (p_data ∩ p_g = ∅) ──► JSD = log 2 (Constant) ──► Zero gradient feedback
  Mode Collapse ──► Generator maps all z to single x* ──► Discriminator detects single mode

```

#### 1. Vanishing Gradients & Non-Saturating Loss:

When the discriminator is optimal ($D(x) \to 1$ for real, $D(G(z)) \to 0$ for fake), the term $\log(1 - D(G(z)))$ saturates early in training, yielding near-zero gradient for $G$:

$$\lim_{D(G(z)) \to 0} \nabla_{\theta_G} \log(1 - D(G(z))) = 0$$

*Heuristic Fix (Non-Saturating Loss):* Modify the generator update to **maximize** $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$:

$$\mathcal{L}_G^{\text{NS}} = -\mathbb{E}_{z \sim p_z}[\log D(G(z))]$$

#### 2. Mode Collapse:

Occurs when the generator optimizes against a local discriminator state by outputting samples from a single high-probability mode of $p_{\text{data}}$, failing to represent the complete diversity of the target distribution.

* **Single-Mode Collapse:** $G(z) = x^*$ for all $z$.
* **Partial Mode Collapse:** $G(z)$ generates only a subset of real classes (e.g., generating only digits '1' and '7' on MNIST).

---

### 5.3 Wasserstein GAN (WGAN & WGAN-GP)

When distributions $p_r$ and $p_g$ lie on lower-dimensional manifolds in high-dimensional space, their overlap is typically set-measure zero. In this scenario, JSD is constant ($\log 2$), providing zero useful gradient to the generator.

#### 1. Earth Mover's (Wasserstein-1) Distance:

Measures the minimum work required to transport probability mass from distribution $p_r$ to $p_g$:

$$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

Using the **Kantorovich-Rubinstein Duality**, this reduces to:

$$W(p_r, p_g) = \sup_{\|f\|_L \le 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{y \sim p_g}[f(y)]$$

where $\|f\|_L \le 1$ represents the class of **1-Lipschitz continuous functions** satisfying $|f(x_1) - f(x_2)| \le |x_1 - x_2|$.

#### 2. WGAN Weight Clipping vs. WGAN-GP (Gradient Penalty):

* **Original WGAN:** Clamps critic weights to a compact interval $w \in [-c, c]$ after each gradient update to enforce 1-Lipschitz continuity.
* *Failure:* Weight clipping forces network weights toward extreme boundaries $\pm c$, causing capacity underutilization or exploding gradients.


* **WGAN with Gradient Penalty (WGAN-GP):** Enforces 1-Lipschitz continuity dynamically by adding a soft penalty term to the critic loss whenever the gradient norm deviates from 1 along interpolated samples $\hat{x} = \epsilon x + (1 - \epsilon) y$ for $\epsilon \sim U(0, 1)$:

$$\mathcal{L}_{\text{Critic}} = \underbrace{\mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})] - \mathbb{E}_{x \sim p_r}[D(x)]}_{\text{Wasserstein Distance Loss}} + \underbrace{\lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ (\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2 \right]}_{\text{Gradient Penalty Term (\lambda = 10)}}$$

---

### 5.4 Specialized GAN Architectures

#### 1. Deep Convolutional GAN (DCGAN) Guidelines:

Radford et al. (2015) established architectural rules for stable convolutional GAN training:

```text
                         DCGAN Architectural Pipeline
                         
  Latent Noise z (100-dim)
            │
            ▼
  Dense Layer + Reshape (4x4x1024)
            │
            ▼ (Fractionally-Strided Conv Transpose 2D + BatchNorm + ReLU)
  Feature Map (8x8x512) ──► (16x16x256) ──► (32x32x128)
            │
            ▼ (ConvTranspose2D + Tanh Activation)
  Generated Synthetic Image (64x64x3)

```

* Replace pooling layers with **strided convolutions** (Discriminator) and **fractionally-strided convolutions** / ConvTranspose2d (Generator).
* Use **Batch Normalization** in both Generator and Discriminator (except Generator output layer and Discriminator input layer).
* Use **ReLU** activation in Generator for all layers except output (which uses **Tanh**).
* Use **LeakyReLU** activation ($\alpha = 0.2$) in Discriminator for all layers.

#### 2. Conditional GAN (cGAN):

Incorporate auxiliary condition vector $y$ (such as class labels or text embeddings) into both Generator and Discriminator:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x, y \sim p_{\text{data}}}[\log D(x \mid y)] + \mathbb{E}_{z \sim p_z, y \sim p_y}[\log (1 - D(G(z \mid y) \mid y))]$$

#### 3. Cycle-Consistent GAN (CycleGAN):

Enables unpaired image-to-image translation between domains $X$ and $Y$ using two generators ($G: X \to Y$, $F: Y \to X$) and two discriminators ($D_Y$, $D_X$).

```text
                      CycleGAN Consistency Mechanism
                      
  Domain X Sample x ──────► Generator G(x) ──────► Domain Y Estimate y_hat
        │                                                │
        │                                                ▼
        └──────────────◄ Generator F(y_hat) ◄────────────┘
                        Reconstructed x_rec ≈ x

```

* **Cycle Consistency Loss:** Guarantees $F(G(x)) \approx x$ and $G(F(y)) \approx y$:

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\Vert{}F(G(x)) - x\Vert{}_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[\Vert{}G(F(y)) - y\Vert{}_1]$$

---

### 5.5 Quantitative Generative Evaluation Metrics

Evaluating generative performance relies on measuring sample quality (fidelity) and sample diversity (coverage).

#### 1. Inception Score (IS):

Evaluates generated samples $x \sim p_g$ by passing them through a pre-trained Inception-v3 classifier:

$$\text{IS}(G) = \exp\left( \mathbb{E}_{x \sim p_g} \left[ \text{KL}(p(y \mid x) \parallel p(y)) \right] \right)$$

* **Sharpness ($p(y \mid x)$):** Conditional class distribution should have low entropy (high confidence).
* **Diversity ($p(y)$):** Marginal distribution $\int p(y \mid x) p_g(x) dx$ should have high entropy (uniform coverage across classes).
* *Limitation:* Insensitive to overfitting; fails if a model memorizes one training sample per class.

#### 2. Fréchet Inception Distance (FID):

Calculates the Wasserstein-2 distance between feature embeddings of real images $r$ and generated images $g$ extracted from the intermediate pool3 layer of Inception-v3:

$$\text{FID} = \Vert{}\mu_r - \mu_g\Vert{}_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ represent the empirical mean vectors and covariance matrices of real and generated activations.

*Lower FID indicates higher visual quality and closer distributional alignment.*

---

## 6. Enterprise Pipeline Architecture

Production GAN pipelines require distinct orchestration for latent sampling, gradient calculation, loss penalty evaluation, and real-time diagnostic reporting.

```text
                     Enterprise Production GAN Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA INGESTION & DATASET SHUFFLING PIPELINE (Real Samples p_data)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ WGAN-GP CRITIC EVALUATION LOOP (n_critic = 5 Iterations per Generator Step)  │
│ • Sample Latent Noise z ~ N(0, I) & Compute Synthetic Images G(z)           │
│ • Calculate Interpolates x_hat = ε·x_real + (1-ε)·x_fake                     │
│ • Evaluate Gradient Penalty Term ||∇_x_hat D(x_hat)||_2                      │
│ • Update Critic Parameters w_D via Adam (β1=0, β2=0.9)                      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GENERATOR OPTIMIZATION LOOP                                                 │
│ • Freeze Critic Parameters & Evaluate -E[D(G(z))]                           │
│ • Update Generator Parameters θ_G via Adam Optimizer                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DIAGNOSTIC REGISTRY & REAL-TIME EVALUATION ENGINE                           │
│ • Calculate Online Fréchet Inception Distance (FID) Batch Scores            │
│ • Monitor Wasserstein Distance Metric Estimate E[D(x_real)] - E[D(x_fake)]  │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Metric & Selection Matrix

| Evaluation Metric | Mathematical Domain | Evaluates Quality (Fidelity) | Evaluates Diversity | Target Vulnerability / Limitation |
| --- | --- | --- | --- | --- |
| **Inception Score (IS)** | KL Divergence of Inception probabilities | Yes | Yes (Class-level only) | Can be gamed by memorizing single images per class |
| **Fréchet Inception Distance (FID)** | Wasserstein-2 Distance in feature space | Yes | Yes | Assumes Gaussian feature distribution in Inception space |
| **Precision (Generative)** | Support boundary overlap | Yes | No | Insensitive to missing modes |
| **Recall (Generative)** | Coverage of real sample distribution | No | Yes | Insensitive to blurry/unrealistic samples |
| **$L_1$ Reconstruction Loss** | Pixel-wise Absolute Error | Yes (Pairwise) | No | Produces blurry outputs due to mean pixel averaging |

---

## 8. Technology & Implementation Matrix

| Module / Package | Key APIs & Functions | Enterprise Capability | Production Best Practice |
| --- | --- | --- | --- |
| **PyTorch (`torch.nn`)** | `nn.ConvTranspose2d`, `nn.utils.spectral_norm` | Custom GAN module building & spectral normalization | Apply `spectral_norm` to discriminator layers for Lipschitz constraint stability. |
| **PyTorch Autograd** | `torch.autograd.grad(outputs, inputs)` | Exact gradient calculation for WGAN-GP penalty | Set `create_graph=True` during gradient penalty computation to allow backprop. |
| **Clean-FID** | `clean_fid.fid.compute_fid()` | Standardized FID evaluation without resizing artifacts | Use clean-fid to eliminate image resizing library discrepancies across frameworks. |
| **TorchVision** | `torchvision.utils.make_grid`, `save_image` | Automated visual tracking during training iterations | Save grid samples at fixed latent vector inputs $z_{\text{fixed}}$ across epochs. |

---

## 9. Personal Understanding

Task 06 details the game-theoretic principles, mathematical proofs, loss mechanics, and architectural innovations of Generative Adversarial Networks.

Key personal insights include:

1. **Adversarial training is a minimax game, not standard optimization:** Minimizing generator loss while maximizing discriminator loss creates dynamic non-convex optimization paths. Without structural constraints, the optimization path can enter unstable limit cycles or experience mode collapse.
2. **Wasserstein distance solves the zero-gradient problem:** Standard JSD reaches a constant ($\log 2$) when probability distributions have disjoint support. Earth Mover's Distance provides continuous, informative gradient signals even when real and fake distributions do not overlap.
3. **Architecture controls stability:** Techniques like **Gradient Penalty (WGAN-GP)**, **Spectral Normalization**, and **Cycle-Consistency** enforce necessary mathematical bounds (e.g., 1-Lipschitz continuity), turning fragile GAN architectures into reliable generative modeling pipelines.

The foundational principle remains:

> **Generative Adversarial Networks frame density estimation as a two-player zero-sum game where non-cooperative minimax dynamics converge to a Nash Equilibrium when the implicit generated probability distribution $p_g$ matches the empirical data distribution $p_{\text{data}}$.**

---

## 10. Interview / Viva Questions

### Q1. Prove that Goodfellow's original GAN minimax value function reduces to $C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)$ when the discriminator is optimal.

**Answer:**

**Step 1:** For a fixed generator $G$, the discriminator optimizes:

$$V(D, G) = \int_{\mathcal{X}} p_{\text{data}}(x) \log D(x) \, dx + \int_{\mathcal{X}} p_g(x) \log (1 - D(x)) \, dx$$

Maximizing the integrand yields optimal discriminator $D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$.

**Step 2:** Substitute $D^*_G(x)$ back into $V(D, G)$:

$$C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[ \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g}\left[ \log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} \right]$$

**Step 3:** Rewrite terms by introducing $\log 2$:

$$C(G) = \mathbb{E}_{x \sim p_{\text{data}}}\left[ \log \left( \frac{1}{2} \cdot \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \right] + \mathbb{E}_{x \sim p_g}\left[ \log \left( \frac{1}{2} \cdot \frac{p_g(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \right]$$

$$C(G) = -\log 2 - \log 2 + \text{KL}\left( p_{\text{data}} \,\Big\Vert{}\, \frac{p_{\text{data}} + p_g}{2} \right) + \text{KL}\left( p_g \,\Big\Vert{}\, \frac{p_{\text{data}} + p_g}{2} \right)$$

By definition of Jensen-Shannon Divergence $\text{JSD}(P \parallel Q) = \frac{1}{2}\text{KL}\left(P \parallel \frac{P+Q}{2}\right) + \frac{1}{2}\text{KL}\left(Q \parallel \frac{P+Q}{2}\right)$:

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)$$

---

### Q2. Explain why the minimax generator loss $\min_G \mathbb{E}_{z}[\log(1 - D(G(z)))]$ causes vanishing gradients, and how the non-saturating loss resolves this issue.

**Answer:**

In early training, the discriminator easily distinguishes real images from crude fake images ($D(G(z)) \to 0$).

Evaluating the gradient of the original minimax generator loss $\mathcal{L}_G = \log(1 - D(G(z)))$ with respect to discriminator output $a = D(G(z))$:

$$\frac{\partial \mathcal{L}_G}{\partial a} = \frac{-1}{1 - a}$$

As $a \to 0$, $\frac{\partial \mathcal{L}_G}{\partial a} \to -1$, providing a flat gradient signal that leads to slow early training.

**Non-Saturating Loss Solution:**

Replacing the loss with $\mathcal{L}_G^{\text{NS}} = -\log D(G(z))$ yields the derivative:

$$\frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial a} = -\frac{1}{a}$$

As $a \to 0$, $\left\vert{} \frac{\partial \mathcal{L}_G^{\text{NS}}}{\partial a} \right\vert{} \to \infty$. This provides strong, informative gradients early in training when generated images are far from real targets.

```text
               Gradient Magnitude comparison (a = D(G(z)))
               
  Discriminator Output a ──► 0.0 (Discriminator wins completely)
  
  Minimax Loss Gradient:      |-1 / (1 - 0)| = 1.0   (Flat/Saturating)
  Non-Saturating Gradient:    |-1 / 0.0|     → ∞     (High signal early)

```

---

### Q3. What is Mode Collapse? Describe three diagnostic symptoms and two structural solutions designed to eliminate it.

**Answer:**

Mode collapse occurs when the generator learns to produce outputs from only a single mode or a small subset of modes in the data distribution, ignoring the remaining variation.

**Diagnostic Symptoms:**

1. **Visual Monotony:** The generator produces near-identical images regardless of input noise vector $z$.
2. **Low Recall / High FID:** The Inception Score may stay high if generated images look sharp, but FID degrades because generated distribution coverage is incomplete.
3. **Discriminator Oscillation:** The discriminator continuously learns to reject one collapsed mode, causing the generator to jump to another single mode in a repeating loop.

**Structural Solutions:**

1. **Wasserstein GAN (WGAN-GP):** Replaces JSD with Earth Mover's Distance, eliminating step-function gradients and stabilizing multi-mode coverage.
2. **Unrolled GANs / Minibatch Discrimination:** Allows the discriminator to examine relationships across multiple batch samples simultaneously, penalizing batches that lack intra-batch variance.

---

### Q4. Detail the Kantorovich-Rubinstein Duality and explain how WGAN converts the intractable Earth Mover's Distance formulation into a computable neural network objective.

**Answer:**

The primal Earth Mover's Distance requires calculating an infimum over all joint distributions $\gamma(x, y) \in \Pi(p_r, p_g)$:

$$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x, y) \sim \gamma}[\Vert{}x - y\Vert{}]$$

This formulation is computationally intractable for high-dimensional continuous neural network distributions.

**Kantorovich-Rubinstein Duality Transformation:**

The dual formulation re-expresses the problem as a supremum over 1-Lipschitz continuous scalar functions $f: \mathcal{X} \to \mathbb{R}$:

$$W(p_r, p_g) = \sup_{\Vert{}f\Vert{}_L \le 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{y \sim p_g}[f(y)]$$

WGAN parameterizes $f$ using a neural network critic $D_w$ with weights $w$. Enforcing 1-Lipschitz constraints ($\Vert{}\nabla_x D_w(x)\Vert{} \le 1$) allows the network to approximate the Wasserstein distance via standard backpropagation:

$$\max_{w \in \mathcal{W}} \mathbb{E}_{x \sim p_r}[D_w(x)] - \mathbb{E}_{z \sim p_z}[D_w(G_\theta(z))]$$

---

### Q5. Compare WGAN Weight Clipping with WGAN Gradient Penalty (WGAN-GP). Why is Gradient Penalty mathematically superior?

**Answer:**

| Characteristic | WGAN Weight Clipping | WGAN Gradient Penalty (WGAN-GP) |
| --- | --- | --- |
| **Constraint Enforcement** | Hard box constraint $w \in [-c, c]$ | Soft penalty term $(\Vert{}\nabla_{\hat{x}} D(\hat{x})\Vert{}_2 - 1)^2$ |
| **Lipschitz Enforcement** | Enforces global bound on weights $w$ | Enforces local 1-Lipschitz bound on gradients $\nabla D$ |
| **Capacity Utilization** | Pushes weights to extreme bounds $\pm c$ | Utilizes full continuous weight representation space |
| **Gradient Stability** | Vulnerable to vanishing or exploding gradients | Maintains smooth gradients near 1 along paths |

```text
               Weight Clipping vs Gradient Penalty Landscapes
               
  Weight Clipping [-c, c]:
  Weights: [ -c <─────────────── 0 ───────────────> +c ]
           Extreme massing at boundaries causes capacity loss.
  
  Gradient Penalty:
  Penalty added whenever ||∇_x D(x)||_2 ≠ 1.
  Keeps gradient norms smooth without constricting weight values.

```

Mathematically, 1-Lipschitz continuity requires that function gradients have a norm of at most 1 ($\Vert{}\nabla f(x)\Vert{} \le 1$). Weight clipping enforces this indirectly by bounding parameter magnitude, which artificially restricts model capacity. WGAN-GP directly penalizes deviations of $\Vert{}\nabla_{\hat{x}} D(\hat{x})\Vert{}_2$ from 1 along straight-line interpolations between real and generated samples, preserving full parameter capacity.

---

### Q6. Formulate the loss function of CycleGAN. Why is Cycle Consistency Loss required for unpaired image-to-image translation?

**Answer:**

In unpaired image-to-image translation, target pairs $(x_i, y_i)$ do not exist. Standard adversarial loss checks if generated images $G(x)$ look like real images in domain $Y$, but it does not guarantee that output $G(x)$ preserves the underlying content of input $x$.

**CycleGAN Full Objective:**

$$\mathcal{L}_{\text{CycleGAN}}(G, F, D_X, D_Y) = \mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) + \mathcal{L}_{\text{GAN}}(F, D_X, Y, X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)$$

**Cycle Consistency Loss Component:**

$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\Vert{}F(G(x)) - x\Vert{}_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[\Vert{}G(F(y)) - y\Vert{}_1]$$

* **Forward Cycle Consistency:** Translating $x \to G(x) \to F(G(x))$ must reconstruct original image $x$.
* **Backward Cycle Consistency:** Translating $y \to F(y) \to G(F(y))$ must reconstruct original image $y$.

Without cycle consistency, generator $G$ could map every image in domain $X$ to a single valid image in domain $Y$ (mode collapse), satisfying the discriminator while destroying source image content.

---

### Q7. Explain the mathematical formulation of Fréchet Inception Distance (FID). What does an FID score of 0 imply?

**Answer:**

Fréchet Inception Distance calculates the Wasserstein-2 distance between two multidimensional Gaussian distributions fitted to feature embeddings from the Inception-v3 pool3 layer:

$$\text{FID}(r, g) = \Vert{}\mu_r - \mu_g\Vert{}_2^2 + \text{Tr}\left( \Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2} \right)$$

* $\mu_r, \mu_g$: Feature mean vectors of real and generated images.
* $\Sigma_r, \Sigma_g$: Feature covariance matrices of real and generated images.
* $\text{Tr}(\cdot)$: Matrix trace operator (sum of diagonal elements).

**Physical Meaning of FID = 0:**

An FID score of 0 implies that feature means and covariances of generated images match real image distributions ($\mu_g = \mu_r$ and $\Sigma_g = \Sigma_r$). Lower FID scores correspond to higher visual quality and closer distributional alignment.

---

### Q8. What is Spectral Normalization in GAN Discriminators, and how does it guarantee 1-Lipschitz continuity?

**Answer:**

Spectral Normalization (Miyato et al., 2018) controls the Lipschitz constant of a neural network by normalizing the weight matrix $W$ of each layer by its matrix $L_2$ operator norm (the largest singular value $\sigma(W)$):

$$W_{\text{SN}} = \frac{W}{\sigma(W)}$$

* **Spectral Norm Definition:** $\sigma(W) = \max_{h \neq 0} \frac{\Vert{}W h\Vert{}_2}{\Vert{}h\Vert{}_2}$, calculated efficiently using Power Iteration.
* **1-Lipschitz Guarantee:** Since the Lipschitz norm of a linear layer $f(h) = W h$ equals $\sigma(W)$, dividing by $\sigma(W)$ forces layer spectral norm to 1.
* By the composition rule of Lipschitz functions, if every layer has a Lipschitz constant $\le 1$, the overall network discriminator $D$ is guaranteed to be 1-Lipschitz continuous ($\Vert{}D\Vert{}_L \le 1$).

---

### Q9. Derive the loss function for a Conditional GAN (cGAN) and explain how auxiliary condition $y$ is incorporated into ConvNet architectures.

**Answer:**

Conditional GAN extends the minimax value function by appending condition vector $y$ to both distributions:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x, y \sim p_{\text{data}}}[\log D(x \mid y)] + \mathbb{E}_{z \sim p_z, y \sim p_y}[\log (1 - D(G(z \mid y) \mid y))]$$

**Architectural Integration:**

1. **Vector Conditioning (Tabular/Dense):** Concatenate target class one-hot vector $y$ directly to latent vector $z$ prior to input: $z_c = [z \,;\, y]$.
2. **Spatial Conditioning (ConvNets):** Replicate label embedding $y$ spatially into a 2D feature map matching image spatial dimensions $(H \times W)$, then channel-wise concatenate it with image tensors:

$$x_c = \text{Concat}_{\text{channel}}(x, \text{SpatialExpand}(y))$$

---

### Q10. What are the core guidelines established by DCGAN for stable deep convolutional GAN architectures?

**Answer:**

Radford et al. defined five architectural rules to stabilize training in convolutional GANs:

1. **Replace Pooling with Strided Convolutions:** Use strided convolutions in the discriminator and fractionally-strided convolutions (transposed convolutions) in the generator.
2. **Use Batch Normalization:** Apply BatchNorm in both generator and discriminator networks to stabilize gradient flow.
3. **Omit BatchNorm in Critical Layers:** Exclude BatchNorm from the generator output layer (to prevent target range distortion) and discriminator input layer (to prevent input sample dependency contamination).
4. **Use LeakyReLU in Discriminator:** Use LeakyReLU activations with slope $\alpha = 0.2$ across all discriminator layers.
5. **Use ReLU and Tanh in Generator:** Use ReLU activations for all hidden generator layers, and bounded Tanh activation for the output layer.

---

### Q11. Compare Inception Score (IS) and Fréchet Inception Distance (FID). What are their respective strengths and limitations?

**Answer:**

| Metric | Primary Inputs | Evaluates Real Data? | Sensitivity to Mode Drops | Core Limitations |
| --- | --- | --- | --- | --- |
| **Inception Score (IS)** | Generated images $G(z)$ only | No (Evaluates generated samples in isolation) | Poor (Insensitive to missing entire classes if generated classes are sharp) | Fails on non-ImageNet domains; easily gamed by memorization |
| **FID** | Real images $X_{\text{real}}$ AND generated images $X_{\text{fake}}$ | Yes (Compares feature distributions) | High (Increases significantly if generated modes are missing) | Requires large sample sizes ($N \ge 10,000$) for stable estimates |

---

### Q12. Explain the difference between Transposed Convolutions and Sub-Pixel Convolutions (PixelShuffle) for Generator upsampling, detailing the checkerboard artifact issue.

**Answer:**

* **Transposed Convolutions (Fractionally-Strided Convolutions):** Insert zeros between spatial features and apply learned convolution kernels.
* *Issue (Checkerboard Artifacts):* Uneven overlap occurs when kernel size is not divisible by stride (e.g., kernel size 3, stride 2), creating repeating grid patterns in generated images.


* **Sub-Pixel Convolution (PixelShuffle):** Applies standard 2D convolutions with output channel dimension $r^2 \cdot C$, then reshapes the tensor space from $(B, r^2 \cdot C, H, W)$ to $(B, C, r \cdot H, r \cdot W)$.
* *Advantage:* Replaces zero-padded transposed convolutions with direct channel reordering, eliminating checkerboard artifacts.



```text
               Checkerboard Artifact Overlap Comparison
               
  Transposed Conv (Kernel 3, Stride 2):
  [ Overlap: 1 ][ Overlap: 2 ][ Overlap: 1 ][ Overlap: 2 ] ──► Checkerboard Grid
  
  PixelShuffle (Standard Conv + Channel Rearrange):
  Standard conv output (B, r^2*C, H, W) ──► Re-arranged directly to (B, C, r*H, r*W)

```

---

### Q13. How does Information-Maximizing GAN (InfoGAN) learn disentangled representations without labeled target data?

**Answer:**

InfoGAN (Chen et al., 2016) splits input noise into unstructured noise $z$ and structured latent codes $c$ (representing specific attributes like rotation, stroke width, or digit identity).

To prevent the generator from ignoring code $c$, InfoGAN adds a **mutual information regularization term** $I(c; G(z, c))$ to the minimax objective:

$$\min_G \max_D V_{\text{InfoGAN}}(D, G) = V(D, G) - \lambda I(c; G(z, c))$$

Since mutual information $I(c; G(z, c)) = H(c) - H(c \mid G(z, c))$ is intractable to compute directly, InfoGAN uses an auxiliary classifier head $Q(c \mid x)$ to optimize a variational lower bound $L_K(G, Q) \le I(c; G(z, c))$, forcing latent code $c$ to correspond to explicit visual variations.

---

### Q14. Describe the fundamental difference between Explicit Density Models (e.g., VAEs) and Implicit Density Models (e.g., GANs).

**Answer:**

| Characteristic | Explicit Density Models (e.g., VAEs, Flow Models) | Implicit Density Models (e.g., GANs) |
| --- | --- | --- |
| **Density Evaluation** | Computes explicit likelihood $p_\theta(x)$ or variational lower bound (ELBO) | Cannot compute explicit scalar likelihood $p(x)$ |
| **Sampling Technique** | Direct sampling from analytical distributions | Samples indirectly by transforming noise $G(z)$ |
| **Loss Paradigm** | Log-likelihood maximization / Reconstruction error | Minimax zero-sum adversarial game |
| **Visual Sharpness** | Tendency toward blurry samples ($L_1/L_2$ pixel averaging) | Produces sharp, realistic images |

---

### Q15. How would you diagnose and recover from Discriminator Over-Powering during GAN training?

**Answer:**

**Diagnosis:**

Discriminator loss rapidly drops to zero ($\mathcal{L}_D \to 0$), while generator loss rises continuously ($\mathcal{L}_G \to \infty$). The discriminator easily identifies fake images, causing generator gradients to vanish.

**Recovery Strategies:**

1. **Apply Spectral Normalization:** Restrict discriminator layer Lipschitz bounds using `nn.utils.spectral_norm`.
2. **Adjust Relative Update Frequencies:** Train the generator for multiple steps per discriminator step (e.g., $k_G = 2, k_D = 1$), or use WGAN-GP where the critic is trained $n_{\text{critic}} = 5$ times per generator update under a continuous gradient signal.
3. **Use Two Time-Scale Update Rule (TTUR):** Set a higher learning rate for the generator than the discriminator (e.g., $\eta_G = 0.0004$, $\eta_D = 0.0001$).
4. **Apply Label Smoothing:** Replace hard target labels ($1.0$ for real) with smoothed targets ($0.9$ for real) to prevent discriminator overconfidence.

---

## 11. Conclusion

Task 06 covers the mathematical foundations of Generative Adversarial Networks, Minimax Two-Player Games, Training Pathologies, Wasserstein Distances, Deep Convolutional Architectures, Conditional Generation, and Image Quality Metrics.

```text
                     GAN Pipeline Execution Pathway
                                   ↓
Data Ingestion & Latent Space Sampling (z ~ N(0, I))
                                   ↓
WGAN-GP Critic Loop (5 Steps: Gradient Penalty ||∇ D(x_hat)||_2 = 1)
                                   ↓
Generator Non-Saturating Optimization Step (-E[D(G(z))])
                                   ↓
Architectural Enforcements (Spectral Normalization / Cycle-Consistency Losses)
                                   ↓
Quantitative Quality Diagnostics (FID & Inception Score Auditing)
                                   ↓
Final Validated Synthetic Generator Model Deployment

```

The core structural pillars of Adversarial Generative Modeling include:

```text
Generative Adversarial Network Pillars
├── Minimax Game Foundations (Zero-Sum Dynamics, Optimal Discriminator D*, JSD Convergence)
├── Training Stability Frameworks (Non-Saturating Loss, WGAN Earth Mover's, WGAN-GP Gradient Penalty)
├── Architectural Innovations (DCGAN Rules, cGAN Class Injection, CycleGAN Cycle-Loss)
└── Diagnostic Evaluation Metrics (Fréchet Inception Distance, Inception Score, Precision/Recall)

```

Core tools and operational frameworks:

```text
PyTorch Framework (nn.ConvTranspose2d, autograd.grad, spectral_norm)
Clean-FID Metric Package (clean_fid.compute_fid)
TorchVision Module (torchvision.utils.make_grid)

```

Completing Task 06 provides the theoretical foundation and practical tools needed to build stable adversarial training loops, apply gradient penalties, resolve mode collapse, implement image-to-image translation models, and quantitatively evaluate synthetic data quality.

The foundational principle remains:

> **Generative Adversarial Networks frame density estimation as a two-player zero-sum game where non-cooperative minimax dynamics converge to a Nash Equilibrium when the implicit generated probability distribution $p_g$ matches the empirical data distribution $p_{\text{data}}$.**

---

## 12. Key Takeaways

1. **Vanilla GANs** optimize a two-player minimax game that minimizes Jensen-Shannon Divergence $\text{JSD}(p_{\text{data}} \parallel p_g)$ when the discriminator is optimal.
2. **Vanishing Gradients** occur when discriminator loss saturates; switching to non-saturating generator loss ($-\log D(G(z))$) provides stronger early gradient signals.
3. **Mode Collapse** causes the generator to output samples from only a few modes; it can be diagnosed via low Recall and high FID scores.
4. **Wasserstein GAN (WGAN)** uses Earth Mover's Distance to maintain continuous gradients even when distributions have disjoint support.
5. **WGAN-GP** enforces 1-Lipschitz continuity by adding a gradient penalty $(\Vert{}\nabla_{\hat{x}} D(\hat{x})\Vert{}_2 - 1)^2$, avoiding the capacity issues of weight clipping.
6. **DCGAN** establishes structural rules (strided convolutions, BatchNorm, LeakyReLU, Tanh) for stable deep convolutional GAN architectures.
7. **Conditional GANs (cGANs)** incorporate condition vectors $y$ into both networks to steer generation toward specific target classes.
8. **CycleGAN** enables unpaired image-to-image translation by combining adversarial loss with Cycle Consistency Loss ($F(G(x)) \approx x$).
9. **Fréchet Inception Distance (FID)** evaluates synthetic image quality and diversity by comparing feature distribution statistics against real data in Inception embedding space.
10. **Spectral Normalization** stabilizes training by dividing layer weights by their largest singular value, guaranteeing 1-Lipschitz discriminator bounds.
