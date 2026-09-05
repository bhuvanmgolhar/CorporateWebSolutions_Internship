# Task 08 — Mathematical Concepts in Data Science: Sampling Distributions, Central Limit Theorem, Probability Theory & Statistical Inference

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VI |
| Task Number | 08 (Capstone Task) |
| Topic | Mathematical Concepts in Data Science: Probability Axioms, Discrete & Continuous Random Variables, Sampling Methods, Sampling Distributions, Central Limit Theorem (CLT), Hypothesis Testing, and Confidence Intervals |
| Task Type | Applied Mathematics, Theoretical Statistics & Core Data Science Foundations |
| Status | Completed |
| Repository Section | `tasks/portal-06/task-08/` |

---

## 2. Objective

The objective of this task is to formalize, analyze, and apply core concepts in **Sampling, Probability Theory, Probability Distributions, and Statistical Inference** as applied to modern Data Science.
This task focuses on:
- Formalizing probability space, Kolmogorov axioms, conditional probability, Bayes' Theorem, and random variables.
- Analyzing foundational discrete and continuous distributions (Binomial, Poisson, Normal, Standard Normal, Student's t, Chi-Square).
- Rigorously proving and applying the **Central Limit Theorem (CLT)** and the **Law of Large Numbers (LLN)** to sample means.
- Implementing probability sampling methodologies (Simple Random, Stratified, Cluster, Systematic) versus non-probability sampling techniques.
- Conducting inferential statistics: Point Estimation, Maximum Likelihood Estimation (MLE), Hypothesis Testing ($Z$-test, $t$-test, ANOVA, Chi-Square), $p$-value interpretation, and Confidence Interval estimation.

---

## 3. Introduction

Probability and sampling form the bedrock of statistical machine learning, predictive modeling, and experimental design. In Data Science, we rarely have access to entire populations; instead, we draw inferences about unknown population parameters $\theta$ using finite observed samples $X = \{x_1, x_2, \dots, x_n\}$. **Probability theory provides the mathematical language to quantify uncertainty, while sampling theory governs how sample statistics generalize to population parameters.**

```text
                  Data Science Statistical Inference Loop
┌─────────────────────────────────────────────────────────────────────────────┐
│ POPULATION DATA (Parameter θ, Mean μ, Variance σ²)                         │
│ (Target universe of interest; usually unobservable or computationally vast) │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         [ Probabilistic Sampling ]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SAMPLE DATA (Statistic θ̂, Sample Mean x̄, Sample Variance s²)               │
│ (Observable subset collected via Stratified / Simple Random Sampling)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         [ Statistical Inference / CLT ]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ESTIMATION & HYPOTHESIS TESTING                                             │
│ Confidence Intervals ──► Hypothesis Testing (p-values) ──► Model Decisions  │
└─────────────────────────────────────────────────────────────────────────────┘

```

The core principle governing data science sampling and probability is:

> **Sampling and probability transform raw, unobserved population spaces into bounded, measurable statistical inferences by grounding sample statistics in the Central Limit Theorem and probability axioms.**

---

## 4. Paradigm Comparison Matrix

Comparing sampling strategies highlights essential trade-offs among selection bias risk, variance reduction, implementation complexity, and representative fidelity.

```text
               Sampling Paradigm Comparison Matrix
┌─────────────────────────┬───────────────────────────────────────────────────┐
│ Sampling Technique      │ Operational Execution & Mathematical Characteristics│
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Simple Random           │ Every unit has an equal inclusion probability     │
│ Sampling (SRS)          │ $P(i) = \frac{n}{N}$; minimal assumptions; higher │
│                         │ sampling variance in heterogeneous populations.    │
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Stratified Random       │ Divides population into non-overlapping strata;    │
│ Sampling                │ samples within strata proportionally; reduces     │
│                         │ variance ($\sigma_{\text{strat}}^2 \le \sigma_{\text{srs}}^2$). │
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Cluster Sampling        │ Divides population into clusters; randomly selects │
│                         │ whole clusters; cost-effective for spatial data;  │
│                         │ higher design effect / variance due to intra-cluster correlation. │
├─────────────────────────┼───────────────────────────────────────────────────┤
│ Systematic Sampling     │ Selects every $k$-th element ($k = \frac{N}{n}$)  │
│                         │ from an ordered frame; efficient; vulnerable to   │
│                         │ periodic patterns or hidden seasonality.          │
└─────────────────────────┴───────────────────────────────────────────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

Understanding statistical inference requires formalizing probability axioms, continuous and discrete distribution metrics, and the Central Limit Theorem.

### 5.1 Probability Formalization & Bayes' Theorem

A probability space is defined by the triple $(\Omega, \mathcal{F}, P)$, where $\Omega$ is the sample space, $\mathcal{F}$ is the event $\sigma$-algebra, and $P: \mathcal{F} \to [0, 1]$ is a probability measure satisfying **Kolmogorov's Axioms**:

1. $P(E) \ge 0, \quad \forall E \in \mathcal{F}$
2. $P(\Omega) = 1$
3. For mutually exclusive events $E_1, E_2, \dots$: $P\left(\bigcup_{i=1}^\infty E_i\right) = \sum_{i=1}^\infty P(E_i)$

#### Conditional Probability & Bayes' Theorem:

Given events $A$ and $B$ where $P(B) > 0$:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)} = \frac{P(B \mid A) P(A)}{\sum_{k} P(B \mid A_k) P(A_k)}$$

Where $P(A)$ is the Prior, $P(B \mid A)$ is the Likelihood, and $P(A \mid B)$ is the Posterior probability.

---

### 5.2 Key Probability Distributions in Data Science

#### 1. Normal (Gaussian) Distribution

For a continuous random variable $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)$$

Standard Normal transformation ($Z$-score): $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$.

#### 2. Binomial Distribution

Models $k$ successes in $n$ independent Bernoulli trials with success probability $p$:

$$P(X = k) = \binom{n}{k} p^k (1 - p)^{n-k}, \quad \mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

#### 3. Poisson Distribution

Models event counts $k$ occurring within a fixed interval at constant rate $\lambda$:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad \mathbb{E}[X] = \lambda, \quad \text{Var}(X) = \lambda$$

---

### 5.3 The Central Limit Theorem (CLT)

Let $X_1, X_2, \dots, X_n$ be independent and identically distributed (i.i.d.) random variables drawn from **any** population distribution with finite mean $\mu$ and finite variance $\sigma^2 > 0$.

As $n \to \infty$, the sample mean $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$ converges in distribution to a Normal distribution:

$$\bar{X}_n \xrightarrow{d} \mathcal{N}\left( \mu, \frac{\sigma^2}{n} \right)$$

Standardized sample mean representation ($Z$-statistic):

$$Z = \frac{\bar{X}_n - \mu}{\frac{\sigma}{\sqrt{n}}} \xrightarrow{d} \mathcal{N}(0, 1)$$

Where $\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$ represents the **Standard Error of the Mean**.

```text
                  Central Limit Theorem Convergence Behavior
 [ Any Non-Normal Distribution ]  ──►  Draw Samples of Size n  ──►  Plot Sample Means x̄
  (Uniform, Exponential, Skewed)            (n = 5, 15, 30+)           (Forms Normal Bell Curve)

```

---

### 5.4 Confidence Intervals & Hypothesis Testing Framework

For a sample mean $\bar{x}$ drawn from a population with unknown mean $\mu$ and known standard deviation $\sigma$:

$$\text{CI}_{1-\alpha} = \bar{x} \pm Z_{\alpha/2} \left( \frac{\sigma}{\sqrt{n}} \right)$$

When population standard deviation $\sigma$ is unknown, we substitute the sample standard deviation $s$ and use Student's $t$-distribution with $df = n - 1$:

$$\text{CI}_{1-\alpha} = \bar{x} \pm t_{\alpha/2, n-1} \left( \frac{s}{\sqrt{n}} \right)$$

---

## 6. Enterprise Data Science Sampling Architecture

Applying theoretical probability and sampling mechanics in enterprise data science platforms relies on automated sampling, hypothesis testing, and inference pipelines.

```text
             Enterprise Probability & Sampling Pipeline
┌─────────────────────────────────────────────────────────────────────────────┐
│ POPULATION DATA STORE (Warehouse / Lakehouse / Spark Dataframe)              │
│ $N = 10^8$ rows; heterogeneous multi-strata data profiles                  │
└──────────────────────┬──────────────────────────────────────┬───────────────┘
                       │                                      │
                       ▼                                      ▼
┌────────────────────────────────────────┐  ┌─────────────────────────────────┐
│ SAMPLING EXECUTION ENGINE              │  │ POWER ANALYSIS & SAMPLE SIZE    │
│ Stratified Key Partitioning            │  │ Calculates minimal $n$ for      │
│ Oversampling / Synthetic (SMOTE)       │  │ Effect Size $d$ & Power $1-\beta$│
└──────────────────────┬─────────────────┘  └─────────────────┬───────────────┘
                       │                                      │
                       ▼                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENTIAL STATISTICAL TESTING & DRIFT HYPOTHESIS ENGINE                    │
│ Two-Sample $t$-test / Chi-Square / KS-Test / ANOVA                          │
└──────────────────────┬──────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE BOUNDS & EXECUTIVE REPORTING                                     │
│ Calculates $95\%$ Confidence Intervals; reports $p$-values and Effect Sizes │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Analysis & Hypothesis Testing Selection Matrix

Selecting the appropriate inferential hypothesis test depends on the variable types, sample independence, and population distribution assumptions.

| Statistical Test | Input Variable Types | Target Outcome Type | Primary Test Statistic | Key Assumptions |
| --- | --- | --- | --- | --- |
| **One-Sample $Z$-Test** | Single continuous feature ($n \ge 30$). | Population mean comparison $\mu_0$. | $Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$ | Known population variance $\sigma$; normal distribution. |
| **One-Sample $t$-Test** | Single continuous feature ($n < 30$). | Population mean comparison $\mu_0$. | $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$ | Unknown $\sigma$; continuous, normally distributed sample data. |
| **Two-Sample Independent $t$-Test** | Categorical (2 groups) + Continuous feature. | Compare means of 2 independent groups. | $t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$ | Independent samples; continuous outputs; equal variances (Welch's $t$-test if unequal). |
| **Paired $t$-Test** | Continuous pre/post measurements. | Mean difference $\bar{d}$ between paired observations. | $t = \frac{\bar{d}}{s_d / \sqrt{n}}$ | Paired/repeated measures; normal difference scores. |
| **One-Way ANOVA** | Categorical ($\ge 3$ groups) + Continuous feature. | Compare means across multiple groups. | $F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}}$ | Homogeneity of variances (Levene's test); normality across groups. |
| **Chi-Square ($\chi^2$) Independence Test** | Two categorical variables. | Evaluate relationship/independence between factors. | $\chi^2 = \sum \frac{(O - E)^2}{E}$ | Independence of observations; expected frequency $E_{ij} \ge 5$ per cell. |

---

## 8. Technology & Integration Matrix

| Functional Role | Industry Standard Tooling | Primary Operational Function |
| --- | --- | --- |
| **Statistical Computation** | SciPy (`scipy.stats`), NumPy | Computes probability density functions, CDFs, standard statistical tests, and $p$-values. |
| **Statistical Modeling** | Statsmodels (`statsmodels`) | Conducts hypothesis tests, ANOVA decomposition, regression diagnostics, and confidence intervals. |
| **Big Data Sampling** | Apache Spark (PySpark SQL), Ray | Executes parallel reservoir sampling, stratified sampling, and distributed dataset splits. |
| **Class Imbalance Sampling** | Imbalanced-Learn (`imblearn`) | Implements SMOTE, ADASYN, and Random Under/Over-sampling strategies for imbalanced datasets. |
| **Visualization & QQ Plots** | Matplotlib, Seaborn | Plots empirical distributions, kernel density estimates (KDE), and Quantile-Quantile (Q-Q) plots. |

---

## 9. Personal Understanding

Task 08 establishes the mathematical foundation required to draw meaningful conclusions from data.
I now realize that **data science is applied statistical inference built on probability theory**. Machine learning models do not operate on fixed, deterministic truths; they estimate conditional probability distributions $P(Y \mid X)$ based on noisy, finite samples drawn from broader populations.
Mastering this mathematical core requires working directly with probability space definitions, sampling distributions, and hypothesis tests. The **Central Limit Theorem** serves as the vital bridge that makes statistical inference possible: regardless of how irregular or skewed a population's underlying distribution may be, the sample mean $\bar{X}$ naturally converges to a normal distribution as sample size grows ($n \ge 30$).
The central principle remains:

> **Sampling and probability transform raw, unobserved population spaces into bounded, measurable statistical inferences by grounding sample statistics in the Central Limit Theorem and probability axioms.**

---

## 10. Interview / Viva Questions

### Q1. What is the Central Limit Theorem (CLT) and why is it fundamental to Data Science?

**Answer:**

The Central Limit Theorem states that if you draw independent, identically distributed (i.i.d.) random samples of size $n$ from **any** population distribution with finite mean $\mu$ and finite variance $\sigma^2$, the sampling distribution of the sample mean $\bar{X}_n$ approaches a Normal distribution $\mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$ as $n \to \infty$ (typically $n \ge 30$). CLT is fundamental because it allows us to construct parametric confidence intervals and conduct hypothesis tests on sample means without needing to know the population's underlying distribution.

### Q2. Explain the difference between probability density functions (PDF) and probability mass functions (PMF).

**Answer:**

* **PMF (Probability Mass Function):** Used for **discrete** random variables $X$. It gives the exact probability that $X$ equals a specific discrete value $x$: $P(X = x) = p(x)$, where $\sum p(x) = 1$.
* **PDF (Probability Density Function):** Used for **continuous** random variables $X$. The probability of taking an exact point value is zero ($P(X = x) = 0$). Instead, the probability that $X$ falls within an interval $[a, b]$ is calculated as the area under the PDF curve: $P(a \le X \le b) = \int_a^b f(x) \, dx$, where $\int_{-\infty}^{\infty} f(x) \, dx = 1$.

### Q3. What is the difference between Simple Random Sampling and Stratified Random Sampling? When should you use Stratified Sampling?

**Answer:**

* **Simple Random Sampling (SRS):** Selects $n$ samples from population $N$ where every individual unit has an equal inclusion probability.
* **Stratified Random Sampling:** Divides the population into mutually exclusive subgroups (strata) based on a key characteristic (e.g., age, income bracket) and draws random samples from each stratum proportionally.
* **When to use:** Use Stratified Sampling when working with heterogeneous populations or imbalanced key subgroups to guarantee representative coverage and reduce sampling variance ($\sigma_{\text{strat}}^2 \le \sigma_{\text{srs}}^2$).

### Q4. Define a $p$-value and correct common misconceptions about its interpretation.

**Answer:**

A **$p$-value** is the probability of obtaining a test statistic at least as extreme as the observed sample value, assuming that the null hypothesis ($H_0$) is true:

$$p\text{-value} = P(\text{Test Statistic as extreme or more} \mid H_0 \text{ is True})$$

* **Common Misconceptions:**
* *False:* $p$-value is the probability that $H_0$ is true.
* *False:* $p$-value is the probability that the experimental result occurred by pure chance.
* *Correct:* A small $p$-value ($p < \alpha$) indicates that observed sample evidence is unlikely under $H_0$, leading us to reject $H_0$.



### Q5. What is the Law of Large Numbers (LLN) and how does it differ from the Central Limit Theorem (CLT)?

**Answer:**

* **Law of Large Numbers (LLN):** States that as sample size $n \to \infty$, the sample mean $\bar{X}_n$ converges in probability to the true population mean $\mu$ ($\bar{X}_n \xrightarrow{p} \mu$). LLN governs **where** the sample mean converges (expected center).
* **Central Limit Theorem (CLT):** Describes the **shape** and spread of the distribution of sample means, showing that $\bar{X}_n$ approaches a Normal distribution with standard error $\frac{\sigma}{\sqrt{n}}$ as $n$ grows.

### Q6. How do Type I ($\alpha$) and Type II ($\beta$) errors relate to Statistical Power in hypothesis testing?

**Answer:**

* **Type I Error ($\alpha$):** Rejecting the Null Hypothesis $H_0$ when it is actually true (False Positive). Alpha is the significance level (e.g., $\alpha = 0.05$).
* **Type II Error ($\beta$):** Failing to reject $H_0$ when it is actually false (False Negative).
* **Statistical Power ($1 - \beta$):** The probability of correctly rejecting a false null hypothesis. Power increases as sample size $n$ increases, as significance level $\alpha$ increases, or as population effect size $d$ increases.

### Q7. What is the mathematical formulation of Bayes' Theorem, and how is it used in probabilistic classification?

**Answer:**

Bayes' Theorem updates the probability of a hypothesis $A$ given observed evidence $B$:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

In Naïve Bayes classification with feature vector $X = (x_1, x_2, \dots, x_n)$ and class label $C_k$:

$$P(C_k \mid X) \propto P(C_k) \prod_{i=1}^n P(x_i \mid C_k)$$

The model assigns the class label maximizing the posterior probability via $\hat{y} = \arg\max_{C_k} P(C_k \mid X)$.

### Q8. What is Standard Error (SE) and how does it differ from Standard Deviation (SD)?

**Answer:**

* **Standard Deviation (SD, $\sigma$ or $s$):** Measures the spread or variability of individual data points around their sample mean within a single dataset.
* **Standard Error (SE, $\text{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}}$):** Measures the variability of the sample mean $\bar{x}$ across repeated sampling iterations from the population. SE quantifies sample mean precision and decreases as sample size $n$ increases.

### Q9. When should you use a Student's $t$-test instead of a $Z$-test?

**Answer:**

Use a **$Z$-test** when population standard deviation $\sigma$ is known AND sample size is large ($n \ge 30$). Use a **Student's $t$-test** when population standard deviation $\sigma$ is unknown (substituted by sample SD $s$) OR when sample size is small ($n < 30$), assuming the underlying sample data is normally distributed.

### Q10. What are parametric versus non-parametric statistical tests?

**Answer:**

* **Parametric Tests (e.g., $t$-test, ANOVA):** Assume underlying sample data follows specific parametric probability distributions (typically Normal) and evaluate parameters like means and variances.
* **Non-Parametric Tests (e.g., Mann-Whitney U, Wilcoxon Signed-Rank, Kruskal-Wallis):** Make no restrictive assumptions about population parameter shapes or distributions. They operate on ranks or medians, making them suitable for skewed or ordinal data.

### Q11. Explain the Chi-Square ($\chi^2$) Test of Independence and its test statistic calculation.

**Answer:**

The Chi-Square test evaluates whether a statistically significant association exists between two categorical variables using contingency tables. The test statistic compares Observed counts ($O_{ij}$) to Expected counts ($E_{ij}$) under the null hypothesis of independence:

$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{\text{Row Total}_i \times \text{Column Total}_j}{\text{Grand Total}}$$

Degrees of freedom: $df = (r - 1)(c - 1)$. If $\chi^2 > \chi^2_{\text{critical}}$, we reject the null hypothesis of independence.

### Q12. What is Maximum Likelihood Estimation (MLE) and how does it estimate model parameters?

**Answer:**

Maximum Likelihood Estimation finds parameter values $\hat{\theta}$ that maximize the likelihood function $L(\theta \mid X)$, which measures how plausible observed sample data $X = \{x_1, \dots, x_n\}$ is under parameter $\theta$:

$$L(\theta \mid X) = \prod_{i=1}^n f(x_i \mid \theta)$$

In practice, we maximize the **Log-Likelihood** function $\ell(\theta) = \ln L(\theta \mid X) = \sum_{i=1}^n \ln f(x_i \mid \theta)$ by taking partial derivatives with respect to $\theta$, setting them to zero ($\frac{\partial \ell}{\partial \theta} = 0$), and solving for $\hat{\theta}$.

### Q13. What is the difference between cluster sampling and stratified sampling?

**Answer:**

* **Stratified Sampling:** Divides population into homogeneous strata based on key features, then samples elements **from every single stratum**. Goal: Reduce sampling variance.
* **Cluster Sampling:** Divides population into naturally occurring heterogeneous clusters (e.g., geographic regions), then randomly samples **entire clusters**, observing all units within selected clusters. Goal: Reduce data collection cost.

### Q14. What is a Quantile-Quantile (Q-Q) plot and how do you interpret it?

**Answer:**

A Q-Q plot is a visual diagnostic tool that compares the empirical quantiles of an observed sample dataset against theoretical quantiles from a reference distribution (typically standard Normal).

* **Interpretation:** If sample data follows the theoretical distribution, points fall closely along a straight 45-degree diagonal line ($y = x$). Deviations or curves away from the line indicate distribution skewness, heavy tails, or kurtosis.

### Q15. How does sample size $n$ impact the width of a Confidence Interval?

**Answer:**

The margin of error for a confidence interval is $ME = Z_{\alpha/2} \left( \frac{\sigma}{\sqrt{n}} \right)$. Because $n$ is in the denominator under a square root:

* Increasing sample size $n$ decreases standard error, **narrowing the confidence interval** and increasing estimation precision.
* To cut the width of a confidence interval in half, sample size $n$ must be quadrupled ($4\times$).

---

## 11. Conclusion

Task 08 provides the statistical and probabilistic foundation required to analyze sample data, evaluate uncertainty, and build sound data science workflows.
The complete statistical inference lifecycle flow is summarized below:

```text
Statistical Inference Lifecycle Flow
      ↓
Population Definition & Probabilistic Sampling Mechanics (Stratified / SRS)
      ↓
Probability Distribution Fitting & Parameter Estimation (MLE)
      ↓
Standard Error Calculation & Central Limit Theorem Convergence Bounds
      ↓
Hypothesis Testing (Z-Test / t-Test / ANOVA / Chi-Square) & p-Value Evaluation
      ↓
Confidence Interval Construction (95% CI Bounds) & Inference Reporting

```

The core structural pillars of Data Science Probability & Sampling include:

```text
Data Science Probability & Sampling Foundations
├── Probability Axioms & Conditional Logic (Kolmogorov Axioms, Bayes' Theorem)
├── Distribution Theory (Normal, Binomial, Poisson, Student's t, Chi-Square)
├── Sampling Mechanics & CLT (SRS, Stratified, Cluster, Standard Error, CLT Limits)
└── Inferential Statistics & Decision Logic (Hypothesis Tests, MLE, p-Values, CIs)

```

Core tools and operational frameworks:

```text
SciPy (scipy.stats) / Statsmodels / NumPy
Apache Spark (PySpark Sampling) / Imbalanced-Learn (SMOTE)
Matplotlib / Seaborn (KDE & Q-Q Plots)

```

By completing Task 08, data scientists master the core mathematical concepts, sampling strategies, probability distributions, and hypothesis tests needed to build reliable, statistically sound predictive models.
The central principle remains:

> **Sampling and probability transform raw, unobserved population spaces into bounded, measurable statistical inferences by grounding sample statistics in the Central Limit Theorem and probability axioms.**

---

## 12. Key Takeaways

1. **Probability theory** provides the mathematical foundation for managing uncertainty in empirical data science.
2. **Kolmogorov Axioms** establish non-negativity, normalization ($P(\Omega)=1$), and additivity for non-overlapping events.
3. **Bayes' Theorem** updates prior beliefs using empirical likelihood to produce posterior probabilities ($P(A \mid B) \propto P(B \mid A) P(A)$).
4. **Simple Random Sampling (SRS)** assigns equal inclusion probabilities to all population elements.
5. **Stratified Sampling** partitions populations into homogeneous groups to reduce sampling variance ($\sigma_{\text{strat}}^2 \le \sigma_{\text{srs}}^2$).
6. The **Central Limit Theorem (CLT)** guarantees that sample means converge to a normal distribution $\mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$ as sample size grows ($n \ge 30$).
7. **Standard Error (SE = $\frac{\sigma}{\sqrt{n}}$)** quantifies the spread of sample means across repeated sampling trials.
8. **Student's $t$-distribution** accounts for extra uncertainty when population standard deviation $\sigma$ is unknown and estimated using sample standard deviation $s$.
9. A **$p$-value** measures the probability of observing a test statistic as extreme as the sample result, assuming $H_0$ is true.
10. **Type I Error ($\alpha$)** occurs when rejecting a true null hypothesis, while **Type II Error ($\beta$)** occurs when failing to reject a false null hypothesis.
11. **Statistical Power ($1 - \beta$)** measures the test's ability to correctly reject a false null hypothesis.
12. **Chi-Square ($\chi^2$) Tests** evaluate independence and goodness-of-fit for categorical variables.
13. **ANOVA ($F$-test)** evaluates mean equality across three or more groups by comparing variance between groups to variance within groups.
14. **Maximum Likelihood Estimation (MLE)** estimates parameter values that maximize the likelihood of observing sample data.
15. **Quantile-Quantile (Q-Q) Plots** visually compare sample quantiles against theoretical normal distributions to check normality assumptions.
