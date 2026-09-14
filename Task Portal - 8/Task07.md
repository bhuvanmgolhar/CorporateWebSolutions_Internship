### Task 07 — Reinforcement Learning (RL), MDPs, Value Functions & Deep RL Algorithms

#### 1. Task Information

| Field | Details |
| --- | --- |
| Internship | Data Science Internship — Portal VIII |
| Task Number | Task 07 (Advanced Reinforcement Learning & Sequential Decision-Making) |
| Topic | Reinforcement Learning Framework: Markov Decision Processes (MDP), Bellman Optimality Equations, Dynamic Programming (Policy & Value Iteration), Temporal Difference Learning (Q-Learning, SARSA), Deep Q-Networks (DQN, Double DQN, Dueling DQN), Policy Gradient Methods (REINFORCE, PPO), Actor-Critic Architectures (A2C/A3C), and Exploration vs. Exploitation Trade-offs ($\epsilon$-greedy, UCB) |
| Task Type | Algorithmic Derivation, Bellman Optimality Proof, & Deep RL Architecture Design |
| Status | Completed |
| Repository Section | `tasks/portal-08/task-07/` |

---

#### 2. Objective

The objective of this task is to provide an exhaustive mathematical, architectural, and empirical analysis of Reinforcement Learning algorithms ranging from tabular decision processes to deep policy gradient methods.

This task covers:

* Formulating sequential decision-making as a Markov Decision Process (MDP) defined by the 5-tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.
* Deriving the Bellman Expectation and Optimality Equations for State-Value $V(s)$ and Action-Value $Q(s, a)$ functions.
* Comparing Model-Based Dynamic Programming (Policy Iteration, Value Iteration) against Model-Free Learning paradigms (Monte Carlo, Temporal Difference).
* Analyzing tabular Model-Free Control algorithms: Off-policy Q-Learning vs. On-policy SARSA.
* Engineering deep value-based architectures: Deep Q-Networks (DQN), addressing overestimation bias with Double DQN (DDQN), and separating state/advantage streams via Dueling DQN.
* Deriving Policy Gradient methods via the Policy Gradient Theorem (REINFORCE) and stabilizing updates using Proximal Policy Optimization (PPO) and Actor-Critic (A2C/A3C) frameworks.

---

#### 3. Introduction & Conceptual Framework

Reinforcement Learning (RL) models sequential decision-making where an autonomous agent learns optimal behavioral strategies through trial-and-error interaction with a dynamic environment to maximize cumulative discounted rewards.

```
                    Reinforcement Learning Interaction Loop
┌─────────────────────────────────────────────────────────────────────────────┐
│                                ENVIRONMENT                                  │
└───────────────┬──────────────────────────────────────────────▲──────────────┘
                │                                              │
     State s_t  │  Reward r_t                                  │ Action a_t
                ▼                                              │
┌──────────────────────────────────────────────────────────────┴──────────────┐
│                                   AGENT                                     │
│  ┌────────────────────────┐                    ┌─────────────────────────┐  │
│  │     Policy π(a|s)      │                    │ Value Function Q(s,a)   │  │
│  └────────────────────────┘                    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

```

The fundamental governing axiom of reinforcement learning is:

> Any sequential decision-making process satisfying the Markov property can be modeled as an MDP where the optimal control policy $\pi^*$ maximizes the expected cumulative discounted return $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$.

---

#### 4. Algorithmic Variants Comparison Matrix

| RL Family | Algorithm | Model Dependency | Optimization Target | On/Off Policy | Key Operational Vulnerability |
| --- | --- | --- | --- | --- | --- |
| Dynamic Programming | Value Iteration | Model-Based | Optimal Value $V^*(s)$ | N/A | Requires explicit transition matrix $\mathcal{P}(s' \mid s, a)$ |
| Model-Free Value | Q-Learning | Model-Free | Action-Value $Q(s, a)$ | Off-Policy | Overestimation bias of Q-values |
| Model-Free Value | SARSA | Model-Free | Action-Value $Q(s, a)$ | On-Policy | Slow convergence due to conservative updates |
| Deep Q-Learning | Double DQN | Model-Free | Deep Neural $Q(s, a; \theta)$ | Off-Policy | Sample inefficient; unstable hyperparameter tuning |
| Policy Gradient | REINFORCE | Model-Free | Parameterized Policy $\pi_\theta(a \mid s)$ | On-Policy | Extremely high variance in gradient estimates |
| Actor-Critic | PPO (Clipped) | Model-Free | Joint Policy $\pi_\theta$ & Value $V_\phi$ | On-Policy | Sensitive to clipping hyperparameter $\epsilon$ |

---

#### 5. Mathematical & Algorithmic Foundations

##### 5.1 Markov Decision Process & Bellman Optimality Formulation

An MDP is formally defined by $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. The return $G_t$ at time step $t$ is:


$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \quad \gamma \in [0, 1)$$

* **Bellman Expectation Equation for $V^\pi(s)$:**

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} \mathcal{P}(s', r \mid s, a) \left[ r + \gamma V^\pi(s') \right]$$


* **Bellman Optimality Equation for $Q^*(s, a)$:**

$$Q^*(s, a) = \sum_{s', r} \mathcal{P}(s', r \mid s, a) \left[ r + \gamma \max_{a'} Q^*(s', a') \right]$$



##### 5.2 Model-Free Control Updates

* **Q-Learning Update (Off-Policy TD Target):**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$


* **SARSA Update (On-Policy TD Target):**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$



##### 5.3 Deep Q-Network Variants

1. **Double DQN Loss Formulation:**
Eliminates overestimation bias by decoupling action selection from action evaluation:

$$\mathcal{L}(\theta) = \mathbb{E}_{\left(s, a, r, s'\right)} \left[ \left( r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right) - Q(s, a; \theta) \right)^2 \right]$$


2. **Dueling Architecture Decomposition:**
Decomposes $Q(s, a)$ into a state-value stream $V(s)$ and an advantage stream $A(s, a)$:

$$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{\vert{}\mathcal{A}\vert{}} \sum_{a'} A(s, a'; \theta, \alpha) \right)$$



##### 5.4 Policy Gradients & Actor-Critic Methods

* **Policy Gradient Theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) \right]$$


* **PPO Clipped Surrogate Objective:**

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$



where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$.

---

#### 6. Enterprise Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ENVIRONMENT INTERACTION & EXPERIENCE REPLAY BUFFER                          │
│ • Parallel Workers collecting transitions (s_t, a_t, r_t, s_{t+1}, d_t)     │
│ • Prioritized Experience Sampling via Segment Trees                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEEP RL OPTIMIZATION ENGINE (Double Dueling DQN / PPO)                       │
│ • Sample Minibatch & Compute Target Values (Target Net θ^- updated periodically)│
│ • Compute Advantage Estimates A_t via Generalized Advantage Estimation (GAE)│
│ • Perform Gradient Step via Adam (Gradient Clipping ||g||_2 <= 0.5)         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ REAL-TIME DIAGNOSTIC MONITORING                                             │
│ • Episode Return Metrics, Average Max Q-Values, Policy Entropy Loss         │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

#### 7. Comparative Metric & Selection Matrix

| Metric / Method | Evaluates Convergence | Handles Continuous Actions | Sample Efficiency | Sensitivity to Hyperparameters |
| --- | --- | --- | --- | --- |
| Deep Q-Networks (DQN) | Via TD-Error Loss | No (Discrete Only) | Medium (Replay Buffer) | High (Learning rate, Target update) |
| Deep Deterministic Policy Gradient (DDPG) | Via Critic Loss | Yes | High (Off-policy) | Extreme (Exploration Noise) |
| Proximal Policy Optimization (PPO) | Via Surrogate Loss | Yes | Medium (On-policy) | Low-Medium (Robust clipping) |
| Soft Actor-Critic (SAC) | Via Entropy-Augmented Q | Yes | High (Off-policy) | Low (Auto-tuned temperature) |

---

#### 8. Technology & Implementation Matrix

| Module / Package | Key APIs & Functions | Enterprise Capability | Production Best Practice |
| --- | --- | --- | --- |
| Stable-Baselines3 | `PPO("MlpPolicy", env)`, `DQN()` | Pre-built, verified Deep RL baseline models | Use vector environments (`SubprocVecEnv`) for scalable multi-worker sampling. |
| PyTorch (`torch.distributions`) | `Categorical`, `Normal` | Custom Policy Gradient & Actor-Critic builds | Compute log probabilities with `.log_prob(action)` directly for autograd backprop. |
| Gymnasium (Farama) | `gym.make()`, `env.step()` | Standardized environment execution loops | Apply wrapper patterns for frame stacking and action space normalization. |

---

#### 9. Personal Understanding

Task 07 details the game-theoretic and dynamic control principles behind Reinforcement Learning algorithms. Key personal insights include:

* **Bootstrapping vs. Sampling:** Temporal Difference (TD) methods bootstrap off current value estimates, reducing update variance at the cost of potential bias, whereas Monte Carlo methods rely on full unrolled trajectories.
* **Overestimation Bias:** Standard Q-learning suffers from systemic overestimation of action-values due to the $\max$ operator; Double DQN resolves this by using two independent network parameter sets for selection and evaluation.
* **Exploration vs. Exploitation:** Balancing exploration ($\epsilon$-greedy, entropy regularization) with exploitation is critical to preventing agents from falling into suboptimal local policy traps.

---

#### 10. Interview / Viva Questions

**Q1. Prove why standard Q-Learning exhibits overestimation bias.**

*Answer:*

Let $Q(s, a)$ be an estimate of the true action-values $V^*(s, a)$ subject to random noise $\epsilon_a$, such that $E[\epsilon_a] = 0$.

The target value in Q-learning uses the max operator: $\mathbb{E}\left[\max_a Q(s, a)\right]$.

By Jensen's Inequality and the convexity of the maximum function:


$$\mathbb{E}\left[\max_a (V^*(s, a) + \epsilon_a)\right] \ge \max_a \mathbb{E}\left[V^*(s, a) + \epsilon_a\right] = \max_a V^*(s, a)$$


Thus, the expected value of the maximum estimated action-value is systematically greater than or equal to the maximum true value, causing overestimation.

**Q2. Explain the difference between On-Policy and Off-Policy learning with examples.**

*Answer:*

* **On-Policy Learning:** The agent evaluates and improves the *same* policy that is used to collect data in the environment. Example: **SARSA**, **PPO**. (The update step accounts for the exploratory actions actually taken).
* **Off-Policy Learning:** The agent evaluates and improves a target policy $\pi(a \mid s)$ while behaving according to a separate behavior policy $\mu(a \mid s)$. Example: **Q-Learning**, **DQN**. (The update assumes optimal greedy action selection regardless of exploration).
