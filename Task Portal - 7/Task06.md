# Task 06 — Descriptive Analysis: Summary Statistics, Data Distributions, Measures of Central Tendency & Dispersion, and Exploratory Data Analysis

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VI |
| Task Number | 06 (Foundational Task) |
| Topic | Descriptive Analysis: Measures of Central Tendency, Measures of Dispersion, Shape Metrics (Skewness & Kurtosis), Exploratory Data Analysis (EDA), Robust Statistics, Correlation Metrics |
| Task Type | Descriptive Statistics, Exploratory Data Analysis & Data Profiling |
| Status | Completed |
| Repository Section | `tasks/portal-06/task-06/` |

---

## 2. Objective

The objective of this task is to establish a mathematical, statistical, and practical framework for performing comprehensive **Descriptive Analysis** and **Exploratory Data Analysis (EDA)** on structured datasets.
This task focuses on:
- Formalizing **Measures of Central Tendency** (Arithmetic Mean, Median, Mode, Geometric Mean, Trimmed Mean) and evaluating their sensitivity to extreme values.
- Quantifying **Measures of Dispersion & Spread** (Variance, Standard Deviation, Interquartile Range, Mean Absolute Deviation, Range).
- Computing distribution **Shape Metrics** (**Skewness** for asymmetry, **Excess Kurtosis** for tail-weight and peakedness).
- Structuring systematic **Exploratory Data Analysis (EDA)** workflows across **Univariate**, **Bivariate**, and **Multivariate** settings.
- Evaluating non-linear and non-parametric correlation estimators (**Pearson $r$**, **Spearman $\rho$**, **Kendall $\tau$**).
- Applying robust outlier detection techniques (**Z-Score**, **Modified Z-Score**, **IQR Outer Fences**) and understanding data visualization edge cases (e.g., **Anscombe's Quartet**, **Simpson's Paradox**).

---

## 3. Introduction

**Descriptive Analysis** forms the foundational pillar of data science. Before constructing complex predictive models or running formal statistical tests, descriptive statistics synthesize, summarize, and characterize the intrinsic properties of an observed dataset without making inferential extrapolations about an unobserved population.

While **Inferential Statistics** uses sample data to deduce underlying population parameters, **Descriptive Statistics** focuses strictly on describing the empirical properties of the sample itself.

```text
               Univariate, Bivariate & Multivariate EDA Workflow
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW DATA STREAM (Numerical & Categorical Features)                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ UNIVARIATE PROFILING                                                        │
│ • Central Tendency: Mean, Median, Mode                                      │
│ • Spread: Variance, Std Dev, IQR                                            │
│ • Shape: Skewness (Asymmetry), Kurtosis (Tails)                             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ BIVARIATE & MULTIVARIATE PROFILING                                          │
│ • Linear / Rank Correlations (Pearson r, Spearman ρ)                        │
│ • Contingency Tables, Cross-Tabulations, Covariance Matrices                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DIAGNOSTIC VISUALIZATION & OUTLIER PROFILING                                │
│ • Histograms, KDE, Boxplots, Violin Plots, Scatter Matrices                 │
│ • Outlier Filtering (IQR Fences, Robust Z-score)                            │
└─────────────────────────────────────────────────────────────────────────────┘

```

The core guiding principle of descriptive data analysis is:

> **Summary statistics condense high-dimensional observations into interpretable metrics, but non-visual summary metrics alone can mask structural anomalies, multi-modality, and non-linear relationships.**

---

## 4. Typology of Data & Applicable Summary Statistics

Understanding data scale and data type determines the valid mathematical operations, central tendency measures, and visualization tools available for analysis.

```text
                       Measurement Scales Hierarchy
                       
       ┌──────────┐ ──► Categories with no intrinsic order (Gender, Color)
       │ Nominal  │     Mode, Contingency Tables, Bar Charts
       └────┬─────┘
            ▼
       ┌──────────┐ ──► Ordered categories without equal intervals (Ratings, Grade)
       │ Ordinal  │     Median, Percentiles, Rank Correlations
       └────┬─────┘
            ▼
       ┌──────────┐ ──► Equal intervals, arbitrary zero point (Temperature °C/°F)
       │ Interval │     Mean, Variance, Pearson Correlation (No ratios)
       └────┬─────┘
            ▼
       ┌──────────┐ ──► Equal intervals, absolute true zero point (Income, Height)
       │  Ratio   │     Geometric Mean, Coefficient of Variation, Ratios valid
       └──────────┘

```

| Scale Level | Mathematical Properties | Central Tendency | Dispersion Metrics | Permissible Visualizations |
| --- | --- | --- | --- | --- |
| **Nominal** | Equality ($=, \neq$) | Mode | Frequency Distribution, Entropy | Bar Plot, Pie Chart |
| **Ordinal** | Comparison ($>, <$) | Median | IQR, Percentiles | Stacked Bar Plot, Boxplot |
| **Interval** | Addition / Subtraction ($+, -$) | Mean, Median | Std Dev, Variance, Range | Histogram, KDE Plot |
| **Ratio** | Multiplication / Division ($\times, \div$) | Geometric Mean, Trimmed Mean | Coefficient of Variation ($CV$) | Scatter Plot, Boxplot, Violin Plot |

---

## 5. Mathematical & Algorithmic Foundations

Formalizing descriptive statistics requires precise equations for location estimators, dispersion metrics, higher-order moments, and correlation coefficients.

### 5.1 Measures of Central Tendency

Central tendency represents the central or typical value for a probability distribution or dataset.

#### 1. Arithmetic Mean ($\mu, \bar{x}$):

The arithmetic average of all observations. Highly sensitive to outliers.

$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$

#### 2. Median ($\tilde{x}$):

The middle score of a dataset sorted in ascending order. Robust to extreme outliers (breakdown point = $50\%$).

$$\tilde{x} = \begin{cases} x_{\left(\frac{n+1}{2}\right)}, & \text{if } n \text{ is odd} \\ \frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2} + 1\right)}}{2}, & \text{if } n \text{ is even} \end{cases}$$

#### 3. Trimmed Mean ($\bar{x}_{\alpha}$):

Calculates the mean after removing a percentage $\alpha$ of the smallest and largest values, balancing efficiency with robustness.

$$\bar{x}_{\alpha} = \frac{1}{n - 2k} \sum_{i=k+1}^{n-k} x_{(i)}, \quad k = \lfloor n \cdot \alpha \rfloor$$

#### 4. Geometric Mean ($G$):

Used for exponentially growing processes, multiplicative growth rates, or percentage changes.

$$G = \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}} = \exp\left( \frac{1}{n} \sum_{i=1}^n \ln(x_i) \right)$$

---

### 5.2 Measures of Dispersion & Spread

Dispersion quantifies the variability, scatter, or spread of data points around the central tendency.

#### 1. Sample Variance ($s^2$) & Bessel's Correction:

Measures squared deviations from the sample mean. $n-1$ is used in the denominator (Bessel's Correction) to ensure an unbiased estimator of population variance $\sigma^2$.

$$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2$$

#### 2. Standard Deviation ($s$):

The square root of variance, restoring units to the original scale of measurement.

$$s = \sqrt{\frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2}$$

#### 3. Interquartile Range ($IQR$):

The range covered by the middle $50\%$ of the data, defined as the difference between the 75th percentile ($Q_3$) and 25th percentile ($Q_1$). Resistant to outliers.

$$IQR = Q_3 - Q_1$$

#### 4. Median Absolute Deviation ($MAD$):

A highly robust measure of dispersion based on median absolute deviations.

$$MAD = \text{median}\left( \vert{}x_i - \tilde{x}\vert{} \right)$$

---

### 5.3 Higher-Order Moments: Skewness & Kurtosis

Higher-order standardized moments characterize distribution shape, asymmetry, and heavy-tailed behavior.

```text
                        Distribution Skewness Visuals
                        
   Negative Skew (Left)         Symmetric (Zero)          Positive Skew (Right)
   Mean < Median < Mode       Mean = Median = Mode        Mode < Median < Mean
      ┌─────────┐                 ┌───┐                     ┌─────────┐
     ╱          │                ╱     ╲                   │          ╲
    ╱           │               ╱       ╲                  │           ╲
  ─┴────────────┴─           ──┴─────────┴──             ──┴────────────┴─

```

#### 1. Skewness ($\gamma_1$ — Third Standardized Moment):

Quantifies the asymmetry of a distribution around its mean.

$$\gamma_1 = \mathbb{E}\left[ \left( \frac{X - \mu}{\sigma} \right)^3 \right] = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^3}{\left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^{3/2}}$$

* $\gamma_1 = 0$: Symmetric distribution (e.g., Standard Normal Distribution).
* $\gamma_1 > 0$: Right-skewed (positive skew); long right tail (e.g., income, house prices).
* $\gamma_1 < 0$: Left-skewed (negative skew); long left tail (e.g., age at death, failure rates).

#### 2. Excess Kurtosis ($\gamma_2$ — Fourth Standardized Moment):

Quantifies the presence of heavy tails and extreme outliers relative to a normal distribution.

$$\gamma_2 = \mathbb{E}\left[ \left( \frac{X - \mu}{\sigma} \right)^4 \right] - 3 = \frac{\frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^4}{\left( \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2 \right)^2} - 3$$

* **Mesokurtic ($\gamma_2 = 0$):** Normal distribution tails.
* **Leptokurtic ($\gamma_2 > 0$):** Heavy tails, high risk of extreme outliers (e.g., financial asset returns).
* **Platykurtic ($\gamma_2 < 0$):** Light tails, fewer extreme outliers (e.g., uniform distribution).

---

### 5.4 Correlation & Association Metrics

Bivariate descriptive metrics evaluate linear and monotonic relationships between variables.

#### 1. Pearson Correlation Coefficient ($r$):

Measures linear relationship strength between continuous variables. Ranges from $-1$ to $+1$.

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

#### 2. Spearman Rank Correlation ($\rho$):

Measures non-linear **monotonic** relationships by ranking observations prior to calculating correlation. Highly robust to non-linear transformations and extreme outliers.

$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$

Where $d_i = \text{rank}(x_i) - \text{rank}(y_i)$.

---

### 5.5 Robust Outlier Detection Formulations

Outliers skew classical statistics (mean, variance). Two standard statistical detection rules isolate anomalies:

1. **IQR Rule (Tukey's Fences):**
* Lower Bound: $Q_1 - 1.5 \times IQR$
* Upper Bound: $Q_3 + 1.5 \times IQR$
* Extreme Outliers: Points beyond $Q_1 - 3.0 \times IQR$ or $Q_3 + 3.0 \times IQR$.


2. **Modified Z-Score ($M_i$):**
Uses median and $MAD$ instead of mean and standard deviation to prevent masked outliers from distorting bounds.
$$M_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{MAD}$$


Observations where $\vert{}M_i\vert{} > 3.5$ are flagged as potential outliers.

---

## 6. Enterprise Data Profiling & EDA Architecture

In modern production systems, exploratory data profiling is automated as an ingestion gate to monitor data drift, schema corruption, and extreme skewness before downstream model consumption.

```text
               Production Data Profiling & Validation Engine
┌─────────────────────────────────────────────────────────────────────────────┐
│ INGESTION DATA STREAM (Batch Parquet / Streaming Kafka)                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AUTOMATED SCHEMA INFERENCE & TYPING                                          │
│ • Detects Continuous, Discrete, Categorical, Datetime, Text types           │
│ • Flags Constant, High-Cardinality, and Null-Heavy Features                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STATISTICAL SUMMARY ENGINE                                                  │
│ • Computes Moments (Mean, Variance, Skew, Kurtosis)                         │
│ • Computes Quantiles (Min, 25%, 50%, 75%, Max) & MAD                        │
│ • Generates Correlation Matrices & Covariance Maps                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OUTLIER & DRIFT MONITORING GATEWAY                                         │
│ • Flags Extreme Skewness (|γ1| > 2) ──► Suggests Log / Box-Cox Transform    │
│ • Flags Extreme Outliers (|Mi| > 3.5) ──► Generates Anomaly Audit Report    │
│ • Compares Population Stability Index (PSI) against Baseline Stats          │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Analysis & Estimator Matrix

| Statistical Metric | Formula Category | Sensitivity to Outliers | Primary Use Case | Limitations |
| --- | --- | --- | --- | --- |
| **Arithmetic Mean** | Location | Extremely High | Symmetric continuous distributions without extreme values. | Heavily distorted by single extreme values. |
| **Median** | Location | Extremely Low (50% breakdown) | Skewed continuous metrics (Income, Latency, Real Estate). | Discards exact numerical differences between values. |
| **Standard Deviation** | Spread | High | Standard Normal metrics, Gaussian hypothesis testing. | Exaggerated by squared distance terms in heavy-tailed data. |
| **IQR** | Spread | Low | Non-normal or skewed data distributions. | Ignores distribution properties outside the 25th-75th percentiles. |
| **Pearson Correlation ($r$)** | Association | High | Assessing straight-line linear trends between features. | Fails completely on non-linear relationships (e.g., quadratic). |
| **Spearman Rank ($\rho$)** | Association | Low | Assessing monotonic curves and ordinal ranked metrics. | Insensitive to magnitude differences due to rank conversion. |
| **Z-Score** | Anomaly | High | Standardized outlier filtering on normal distributions. | Mean and Std Dev are distorted by the outliers being detected. |
| **Modified Z-Score ($M_i$)** | Anomaly | Low | Robust outlier filtering on skewed or noisy data. | Less familiar in non-technical business contexts. |

---

## 8. Personal Understanding

Task 06 reinforces the principle that **data science begins with rigorous descriptive profiling, not model fitting**.

I now recognize that relying solely on high-level statistical summaries (such as the mean and standard deviation) can lead to inaccurate conclusions if the underlying distribution is multi-modal, highly skewed, or corrupted by heavy-tailed outliers. A prime example is **Anscombe's Quartet**, where four distinct datasets share identical numerical means, variances, correlations, and linear regression lines, yet display completely different patterns when plotted.

Key personal insights include:

1. **Selecting the right metric depends on distribution shape:** Using the arithmetic mean for right-skewed data (like software latency or income) provides a misleading representation of central tendency; median or trimmed mean must be used instead.
2. **Standard deviation can be misleading in heavy-tailed environments:** In financial returns or network traffic, excess kurtosis ($\gamma_2 > 0$) causes variance to be dominated by rare, extreme events. Metrics like $IQR$ or $MAD$ offer a more realistic measure of typical variability.
3. **Visualization is essential alongside summary metrics:** Combining quantitative metrics with KDE plots, histograms, and boxplots provides a complete view of dataset geometry before engineering features or building models.

The central guiding principle remains:

> **Summary statistics condense high-dimensional observations into interpretable metrics, but non-visual summary metrics alone can mask structural anomalies, multi-modality, and non-linear relationships.**

---

## 9. Interview / Viva Questions

### Q1. What is the fundamental difference between Descriptive and Inferential Statistics?

**Answer:**

* **Descriptive Statistics** summarizes, organizes, and describes the observed features of a specific sample dataset using summary metrics (mean, median, standard deviation) and visual representations (histograms, scatter plots) without making broader claims.
* **Inferential Statistics** uses sample data to draw probabilistic conclusions, estimate unknown parameters, and test hypotheses about an unobserved broader population using probability theory, confidence intervals, and hypothesis testing ($p$-values, $t$-tests).

### Q2. Why is the Median considered a "robust" estimator compared to the Arithmetic Mean?

**Answer:**

The median has a **breakdown point of $50\%$**, meaning up to $50\%$ of the data points can be corrupted or shifted to infinity without driving the median to infinity. Conversely, the arithmetic mean has a **breakdown point of $0\%$** ($1/n$), because a single extreme outlier can arbitrarily distort the mean score.

### Q3. Explain the relationship between Mean, Median, and Mode in Positively and Negatively Skewed distributions.

**Answer:**

* **Positively Skewed (Right-skewed):** The tail extends towards higher values. The extreme values pull the mean upward:
$$\text{Mode} < \text{Median} < \text{Mean}$$


* **Symmetric Distribution:** The measures coincide at the center:
$$\text{Mean} = \text{Median} = \text{Mode}$$


* **Negatively Skewed (Left-skewed):** The tail extends towards lower values. Extreme low values pull the mean downward:
$$\text{Mean} < \text{Median} < \text{Mode}$$



### Q4. What is Excess Kurtosis, and what does a Leptokurtic distribution imply for risk management?

**Answer:**

**Excess Kurtosis** compares a distribution's kurtosis to that of a normal distribution ($\text{Kurtosis} = 3$): $\gamma_2 = \text{Kurtosis} - 3$.

A **Leptokurtic distribution** ($\gamma_2 > 0$) features a sharper central peak and **fat, heavy tails**. In risk management (e.g., quantitative finance), a leptokurtic distribution implies that extreme, high-impact events ("black swan" market crashes or extreme server latency spikes) occur with much higher probability than predicted by a standard Gaussian model.

### Q5. What is Bessel's Correction, and why do we divide sample variance by $n-1$ instead of $n$?

**Answer:**

When calculating sample variance, using $\bar{x}$ (the sample mean) instead of $\mu$ (the true population mean) understates overall variance, because sample data points are naturally closer to their own sample mean than to the true population mean. Dividing by $n$ produces a **biased estimator** that systematically underestimates population variance. Dividing by $n-1$ (**Bessel's Correction**) corrects this downward bias, making the sample variance $s^2$ an **unbiased estimator** of $\sigma^2$:

$$\mathbb{E}[s^2] = \sigma^2$$

### Q6. How does the Interquartile Range (IQR) method establish upper and lower outlier fences?

**Answer:**

Tukey's IQR method defines standard non-outlier limits based on quartiles:

* $\text{Lower Fence} = Q_1 - 1.5 \times IQR$
* $\text{Upper Fence} = Q_3 + 1.5 \times IQR$
Where $IQR = Q_3 - Q_1$. Any data point falling outside these bounds is classified as a mild outlier. Values beyond $Q_1 - 3.0 \times IQR$ or $Q_3 + 3.0 \times IQR$ are flagged as extreme outliers. This method is distribution-free and resistant to extreme values.

### Q7. What is the key difference between Pearson $r$ and Spearman $\rho$ correlation?

**Answer:**

* **Pearson $r$** measures the strength and direction of a strictly **linear relationship** between two continuous variables using raw values.
* **Spearman $\rho$** measures the strength and direction of a **monotonic relationship** by applying Pearson's formula to the **ranked values** of the variables. Spearman can capture non-linear monotonic relationships (e.g., exponential growth) and is resistant to extreme outliers.

### Q8. What is Anscombe's Quartet, and what key lesson does it teach data scientists?

**Answer:**

**Anscombe's Quartet** consists of four artificial datasets created by statistician Francis Anscombe in 1973. All four datasets have nearly identical summary statistics (mean of $x$, mean of $y$, variance of $x$, variance of $y$, correlation between $x$ and $y$, and linear regression line). However, when plotted visually:

* Dataset 1 is a clean linear trend.
* Dataset 2 is a non-linear quadratic curve.
* Dataset 3 is a linear trend corrupted by a single outlier.
* Dataset 4 shows a vertical cluster with an influential point.
**Lesson:** Summary statistics can hide critical structural features; **data visualization is required alongside quantitative profiling**.

### Q9. Compare Nominal, Ordinal, Interval, and Ratio scales of measurement.

**Answer:**

* **Nominal:** Categorical identifiers without intrinsic ranking (e.g., zip codes, eye color). Operations: equality ($=, \neq$).
* **Ordinal:** Ranked categories with unequal or undefined spacing (e.g., survey satisfaction levels: Poor, Fair, Good). Operations: order ($>, <$).
* **Interval:** Ordered numerical values with equal distances between units, but **no absolute zero** (e.g., Temperature in °C or °F). Ratios are invalid ($20^\circ\text{C}$ is not "twice as hot" as $10^\circ\text{C}$).
* **Ratio:** Numerical values with equal distances and an **absolute zero point** representing complete absence (e.g., Income, Weight, Distance). Ratios are fully valid ($20\text{ kg}$ is twice as heavy as $10\text{ kg}$).

### Q10. What is the Mean Absolute Deviation (MAD), and when is it preferred over Standard Deviation?

**Answer:**

**MAD** measures the average distance between each data point and the median: $\text{MAD} = \text{median}(\vert{}x_i - \tilde{x}\vert{})$.

Unlike standard deviation, which squares deviations (magnifying extreme values), MAD measures absolute linear deviations from the median. It is preferred when analyzing noisy, non-Gaussian, or heavy-tailed datasets where standard deviation is skewed by extreme outliers.

### Q11. How can you identify multi-modality in a dataset using descriptive statistics?

**Answer:**

Multi-modality (multiple peaks) can be detected through:

1. **Discrepancy between Mean/Median and Multiple Local Modes:** Discrete frequency tables show multiple peak counts.
2. **Kurtosis Anomalies:** Negative excess kurtosis ($\gamma_2 < 0$, platykurtic behavior) often signals a flat or multi-peaked top.
3. **Kernel Density Estimation (KDE) / Histograms:** Visual inspection displays distinct peaks separated by troughs.
4. **Statistical Tests:** Formal tests like **Hartigan's Dip Test** evaluate whether a distribution significantly departs from unimodality.

### Q12. What is the Empirical Rule (68-95-99.7 Rule) for Gaussian distributions?

**Answer:**

For a perfectly normal (Gaussian) distribution:

* Approximately **$68.27\%$** of observations fall within $\mu \pm 1\sigma$.
* Approximately **$95.45\%$** of observations fall within $\mu \pm 2\sigma$.
* Approximately **$99.73\%$** of observations fall within $\mu \pm 3\sigma$.
If empirical sample data deviates significantly from these proportions, the underlying distribution is non-normal.

### Q13. What is Covariance, and why do we standardize it into Correlation?

**Answer:**

**Covariance** measures the joint variability of two random variables:

$$\text{Cov}(X, Y) = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{n-1}$$

However, covariance is **scale-dependent**; its magnitude depends on the units of measurement (e.g., covariance in meters is $1000\times$ larger than in kilometers). Standardizing covariance by dividing by the product of both standard deviations ($\sigma_X \cdot \sigma_Y$) yields the **Pearson Correlation Coefficient ($r$)**, a scale-free metric bounded between $-1$ and $+1$.

### Q14. Explain Simpson's Paradox and how it manifests in descriptive analysis.

**Answer:**

**Simpson's Paradox** occurs when a trend or association observed across multiple subgroups **reverses direction** when the subgroups are combined into an aggregated whole. It typically occurs when a hidden confounding variable influences both group assignment and outcome metrics. It highlights the risk of relying solely on aggregated descriptive statistics without evaluating relevant conditional subgroups.

### Q15. How do missing data mechanisms (MCAR, MAR, MNAR) impact descriptive statistics?

**Answer:**

* **Missing Completely at Random (MCAR):** Missingness is independent of both observed and unobserved data. Removing missing rows decreases sample size and efficiency, but does **not bias** central tendency or dispersion estimators.
* **Missing at Random (MAR):** Missingness depends systematically on observed features (e.g., older respondents skipping income questions). Removing missing values creates **selection bias**; conditional imputation is required to restore accurate descriptive stats.
* **Missing Not at Random (MNAR):** Missingness depends on the unobserved missing value itself (e.g., individuals with high debt choosing not to report debt). Simple summary statistics calculated on observed data are fundamentally **biased**, requiring sensitivity analysis or pattern-mixture modeling.

---

## 10. Conclusion

Task 06 establishes the core methodologies for Descriptive Analysis and Exploratory Data Analysis.

```text
Exploratory Data Analysis (EDA) Execution Flow
      ↓
Dataset Ingestion & Schema Inspection (Categorical vs Continuous)
      ↓
Univariate Profiling (Location, Spread, Skewness & Excess Kurtosis)
      ↓
Outlier Auditing (IQR Fences & Modified Z-Scores)
      ↓
Bivariate Association (Pearson r, Spearman ρ & Cross-Tabulations)
      ↓
Visual Diagnostics (KDE, Boxplots, Violin Plots & Scatter Matrices)

```

The core structural pillars of Descriptive Analysis include:

```text
Descriptive Analysis Pillars
├── Central Tendency (Arithmetic Mean, Median, Trimmed Mean, Mode)
├── Measures of Spread (Variance, Standard Deviation, IQR, MAD)
├── Distribution Geometry (Skewness Asymmetry, Kurtosis Tail-Weight)
└── Association & Diagnostics (Pearson r, Spearman ρ, Outlier Filtering)

```

Core tools and operational frameworks:

```text
Pandas / NumPy (Fast vectorised summary statistics, quantile scoring, covariance)
SciPy Stats (Skewness, Kurtosis, Trimmed Means, Spearman/Kendall calculations)
Seaborn / Matplotlib (Histograms, KDE plots, Boxplots, Pairplots, Heatmaps)
ydata-profiling / Sweetviz (Automated exploratory data profiling reports)

```

By mastering Task 06, data scientists gain the analytical foundation required to audit data quality, uncover hidden structures, handle heavy-tailed features, and prepare robust feature pipelines for downstream machine learning.

The central guiding principle remains:

> **Summary statistics condense high-dimensional observations into interpretable metrics, but non-visual summary metrics alone can mask structural anomalies, multi-modality, and non-linear relationships.**

---

## 11. Key Takeaways

1. **Descriptive Statistics** summarize observed sample properties without making probabilistic population inferences.
2. **Arithmetic Mean** is sensitive to extreme values, whereas **Median** provides a robust breakdown point ($50\%$).
3. **Trimmed Mean** offers a balance between central efficiency and outlier robustness.
4. **Bessel's Correction ($n-1$)** removes downward bias when estimating population variance from sample data.
5. **Standard Deviation** measures variance in original data units, but can be distorted by heavy-tailed data.
6. **IQR ($Q_3 - Q_1$)** provides a robust measure of spread that captures the middle $50\%$ of observations.
7. **Median Absolute Deviation (MAD)** offers an outlier-resistant alternative to standard deviation.
8. **Skewness ($\gamma_1$)** quantifies distribution asymmetry; right-skewed distributions have $\text{Mean} > \text{Median}$.
9. **Excess Kurtosis ($\gamma_2$)** measures tail weight; **Leptokurtic** distributions ($\gamma_2 > 0$) indicate fat tails and higher outlier risks.
10. **Pearson Correlation ($r$)** measures linear trends, whereas **Spearman Rank ($\rho$)** captures non-linear monotonic relationships.
11. **Tukey's IQR Fences** ($1.5 \times IQR$) isolate outliers without relying on normal distribution assumptions.
12. **Modified Z-Scores** use median and $MAD$ to detect anomalies in skewed or noisy datasets.
13. **Anscombe's Quartet** demonstrates that visually identical summary metrics can mask vastly different data shapes.
14. **Simpson's Paradox** shows that aggregated trends can reverse when data is separated into subgroups.
15. **Exploratory Data Analysis (EDA)** must combine quantitative metrics with visual diagnostics before building downstream predictive models.
