# Task 01 — Time-Series Data & Forecasting: Classical Decomposition, Stationarity Dynamics, Stochastic Processes (ARIMA/SARIMA), Machine Learning Pipelines & Deep Sequential Architectures

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VIII |
| Task Number | Task 01 (Foundational Task) |
| Topic | Time-Series Analysis & Forecasting: Temporal Properties, Stationarity Transformations (ADF/KPSS), Classical Decomposition, Stochastic Autoregressive Models (AR, MA, ARMA, ARIMA, SARIMA), Machine Learning Feature Engineering (Lags, Rolling Stats), Deep Sequential Architectures (LSTM, GRU, Temporal Convolutional Networks, Transformers), and Temporal Validation Metrics |
| Task Type | Time-Series Dynamics, Temporal Modeling & Sequential Signal Analysis |
| Status | Completed |
| Repository Section | `tasks/portal-08/task-01/` |

---

## 2. Objective

The objective of this task is to provide an in-depth mathematical, statistical, and algorithmic analysis of **Time-Series Analysis and Sequential Forecasting Engine Architectures**.
This task covers:
- Dissecting time-series components via **Additive and Multiplicative Classical Structural Decomposition** (Trend, Seasonality, Cyclic, Irregular/Noise).
- Formulating **Stationarity Dynamics** and statistical hypothesis testing using **Augmented Dickey-Fuller (ADF)** and **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)** tests.
- Proving autoregressive stochastic processes: **Auto-Regressive (AR)**, **Moving Average (MA)**, **ARMA**, **ARIMA**, and **Seasonal ARIMA (SARIMA)** models.
- Determining model orders $(p, q)$ and $(P, Q)$ using **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** correlograms.
- Engineering supervised machine learning features for temporal tabular models (**Lag Features**, **Rolling/Expanding Window Aggregations**, **Calendar/Cyclical Trigonometric Encodings**).
- Analyzing deep learning architectures for sequential modeling: **Recurrent Neural Networks (RNN)**, **Long Short-Term Memory (LSTM)**, **Gated Recurrent Units (GRU)**, **Temporal Convolutional Networks (TCN)**, and **Temporal Transformers**.
- Establishing leakage-free temporal validation strategies (**Time-Series Split**, **Expanding/Rolling Window Cross-Validation**) and evaluation metrics (**RMSE**, **MAE**, **MAPE**, **SMAPE**, **MASE**).

---

## 3. Introduction

A **Time Series** is an ordered sequence of real-valued observations $X = \{x_1, x_2, \dots, x_T\}$ recorded at uniform discrete temporal intervals $t \in \{1, 2, \dots, T\}$. Unlike standard cross-sectional data, time-series data exhibits **temporal dependence** (autocorrelation), meaning observations at step $t$ depend on prior states $t-k$.

```text
                 Time-Series Forecasting Architectural Framework
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW TEMPORAL DATASTREAM X = {x_1, x_2, ..., x_T}                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌─────────────────────────┐                           ┌───────────────────────┐
│ STRUCTURAL DECOMPOSITION│                           │ STATIONARITY TESTING  │
│ • Trend (T_t)           │                           │ • ADF Test (Unit Root)│
│ • Seasonality (S_t)     │                           │ • KPSS Test (Trend)   │
│ • Noise / Residual (I_t)│                           │ • Differencing (Δ^d)  │
└───────────┬─────────────┘                           └───────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FORECASTING MODEL SELECTION & FORMULATION                                   │
│ ┌────────────────────────┐ ┌────────────────────────┐ ┌───────────────────┐ │
│ │ Statistical (SARIMA)   │ │ Machine Learning (GBDT)│ │ Deep Learning     │ │
│ │ • ACF / PACF Analysis  │ │ • Lags / Rolling Stats │ │ • LSTM / GRU / TCN│ │
│ └───────────┬────────────┘ └───────────┬────────────┘ └─────────┬─────────┘ │
└─────────────┼──────────────────────────┼────────────────────────┼───────────┘
              └──────────────────────────┼────────────────────────┘
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEMPORAL CROSS-VALIDATION & OUT-OF-SAMPLE AUDIT                             │
│ • Expanding / Rolling Window Validation (Strict Temporal Ordering)          │
│ • Error Metrics: RMSE, MAE, MAPE, SMAPE, MASE                               │
└─────────────────────────────────────────────────────────────────────────────┘

```

The fundamental governing axiom of time-series analysis is:

> **Time-series modeling exploits temporal autocorrelation, structural trends, and recurring seasonal patterns to construct mapping functions from historical states to future horizons while maintaining strict temporal ordering to prevent data leakage.**

---

## 4. Algorithmic Family Comparison Matrix

Choosing between statistical, machine learning, and deep sequential models depends on sequence length, seasonal complexity, feature dimensionality, and dataset volume.

```text
                   Time-Series Modeling Taxonomy Matrix
┌─────────────┬──────────────────────────┬───────────────────────┬────────────┐
│ Model       │ Underlying               │ Temporal Domain &     │ Data /     │
│ Family      │ Formulation              │ Structural Capability │ Computational│
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Statistical │ Autoregressive linear    │ Univariate linear     │ Low Data / │
│ (ARIMA /    │ stochastic differential  │ dependencies; strictly│ Extremely  │
│ SARIMA)     │ equations (stationary)   │ linear forecasting    │ Fast CPU   │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Machine     │ Supervised tabular       │ Complex non-linear    │ Medium Data│
│ Learning    │ GBDT / RF using lag and  │ feature interactions; │ Fast GPU / │
│ (XGB/LGBM)  │ rolling window statistics│ handles exogenous X   │ Multi-CPU  │
├─────────────┼──────────────────────────┼───────────────────────┼────────────┤
│ Deep        │ Gated Recurrent Units    │ Non-linear long-range │ High Data /│
│ Learning    │ (LSTM/GRU) / Temporal    │ temporal dependencies │ Requires   │
│ (LSTM/TCN)  │ Convolutions / Attention │ & multivariate paths  │ Modern GPU │
└─────────────┴──────────────────────────┴───────────────────────┴────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

A rigorous understanding of time-series analysis requires formalizing temporal decomposition, stationarity constraints, stochastic autoregression, feature engineering transformations, and temporal validation metrics.

### 5.1 Classical Structural Decomposition

A time-series observation $y_t$ at time $t$ is decomposed into three underlying components:

1. **Trend-Cycle ($T_t$):** Long-term directional movement or underlying growth trajectory.
2. **Seasonality ($S_t$):** Fixed, recurring periodic fluctuations occurring at known intervals (e.g., daily, weekly, annual).
3. **Irregular/Noise ($I_t$ or $\epsilon_t$):** Unpredictable random residual noise assumed to follow white noise $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$.

```text
                Classical Time-Series Components Scheme
                
   y_t ▲   Original Series: Observed Signal
       │   /\    /\    /\    /\    /\    /\    /\
       └──/──\──/──\──/──\──/──\──/──\──/──\──/──\────────► Time (t)
   
   T_t ▲   Trend Component: Smooth Directional Path
       │                           /───────────────
       │               /───────────
       └──────────────/───────────────────────────► Time (t)
   
   S_t ▲   Seasonal Component: Fixed Periodic Oscillations
       │   /\    /\    /\    /\    /\    /\    /\
       └──/──\──/──\──/──\──/──\──/──\──/──\──/──\────────► Time (t)

```

#### 1. Additive Decomposition Model:

Used when the variance of seasonal fluctuations remains constant regardless of the overall series level.

$$y_t = T_t + S_t + I_t$$

#### 2. Multiplicative Decomposition Model:

Used when seasonal variations scale proportionally with changes in the overall trend level.

$$y_t = T_t \times S_t \times I_t$$

*Logarithmic Transformation Link:* Taking the natural logarithm converts a multiplicative model into an additive model:

$$\ln(y_t) = \ln(T_t) + \ln(S_t) + \ln(I_t)$$

---

### 5.2 Stationarity Dynamics & Hypothesis Testing

A time series is **Strictly Stationary** if its joint probability distribution is invariant to temporal shifts. In practice, models rely on **Weak (Second-Order) Stationarity**:

1. **Constant Mean:** $\mathbb{E}[y_t] = \mu \quad \forall t$
2. **Constant Variance:** $\text{Var}(y_t) = \sigma^2 < \infty \quad \forall t$
3. **Autocovariance depends only on lag $k$:** $\text{Cov}(y_t, y_{t-k}) = \gamma(k)$, independent of absolute time $t$.

#### Statistical Tests for Stationarity:

| Test Name | Null Hypothesis ($H_0$) | Alternative Hypothesis ($H_1$) | Goal for Stationarity |
| --- | --- | --- | --- |
| **Augmented Dickey-Fuller (ADF)** | Series has a **Unit Root** (Non-Stationary) | Series is Stationary | Reject $H_0$ ($p < 0.05$) |
| **KPSS Test** | Series is **Trend-Stationary** | Series has a Unit Root (Non-Stationary) | Fail to reject $H_0$ ($p > 0.05$) |

#### Mathematical Differencing Transformation:

If a series is non-stationary, order-$d$ differencing ($\Delta^d$) removes trend unit roots:

$$\Delta y_t = y_t - y_{t-1} = (1 - B) y_t$$

$$\Delta^2 y_t = \Delta(\Delta y_t) = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) = (1 - B)^2 y_t$$

Where $B$ is the **Backshift (Lag) Operator**: $B^k y_t = y_{t-k}$.

---

### 5.3 Stochastic Autoregressive Models (AR, MA, ARMA, ARIMA, SARIMA)

#### 1. Auto-Regressive $\text{AR}(p)$ Model:

Models $y_t$ as a linear combination of its past $p$ values:

$$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \dots + \phi_p y_{t-p} + \epsilon_t$$

#### 2. Moving Average $\text{MA}(q)$ Model:

Models $y_t$ as a linear combination of current and past $q$ white noise forecast errors:

$$y_t = c + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j} = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}$$

#### 3. $\text{ARMA}(p, q)$ Model:

Combines $\text{AR}(p)$ and $\text{MA}(q)$ for stationary series:

$$y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t + \sum_{j=1}^q \theta_j \epsilon_{t-j}$$

Using the backshift operator $B$:

$$\left(1 - \sum_{i=1}^p \phi_i B^i\right) y_t = c + \left(1 + \sum_{j=1}^q \theta_j B^j\right) \epsilon_t \implies \Phi_p(B) y_t = c + \Theta_q(B) \epsilon_t$$

#### 4. $\text{ARIMA}(p, d, q)$ Model:

Incorporates non-seasonal differencing order $d$ to handle non-stationary series:

$$\Phi_p(B) (1 - B)^d y_t = c + \Theta_q(B) \epsilon_t$$

#### 5. Seasonal ARIMA — $\text{SARIMA}(p, d, q) \times (P, D, Q)_s$ Model:

Extends ARIMA to handle seasonal patterns of period $s$:

$$\Phi_p(B) \tilde{\Phi}_P(B^s) (1 - B)^d (1 - B^s)^D y_t = \Theta_q(B) \tilde{\Theta}_Q(B^s) \epsilon_t$$

Where:

* $p, d, q$: Non-seasonal AR, Differencing, and MA orders.
* $P, D, Q$: Seasonal AR, Differencing, and MA orders.
* $s$: Seasonal period multiplier (e.g., $s=12$ for monthly, $s=7$ for daily data).

#### Model Order Identification via Correlograms (ACF / PACF):

```text
                    ACF & PACF Order Selection Rules
┌────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Model          │ Autocorrelation (ACF)        │ Partial Autocorrelation(PACF)│
├────────────────┼──────────────────────────────┼──────────────────────────────┤
│ AR(p)          │ Tails off exponentially      │ Cuts off sharply after lag p │
│ MA(q)          │ Cuts off sharply after lag q │ Tails off exponentially      │
│ ARMA(p, q)     │ Tails off exponentially      │ Tails off exponentially      │
└────────────────┴──────────────────────────────┴──────────────────────────────┘

```

---

### 5.4 Supervised ML Feature Engineering for Temporal Data

To use supervised algorithms (e.g., XGBoost, LightGBM) on time-series data, sequence data must be converted into a tabular feature matrix $\mathbf{X} \in \mathbb{R}^{N \times k}$ with target vector $\mathbf{y} \in \mathbb{R}^N$.

```text
               Supervised Sliding Window Feature Conversion
               
  Time Series: [y_1, y_2, y_3, y_4, y_5, y_6]  ---> Sliding Window (size=3)
  
  ┌─────────────────────────┬──────────┐
  │ Features (X)            │ Target(y)│
  ├─────────────────────────┼──────────┤
  │ Lag_3   Lag_2   Lag_1   │ Target   │
  ├─────────────────────────┼──────────┤
  │  y_1     y_2     y_3    │   y_4    │
  │  y_2     y_3     y_4    │   y_5    │
  │  y_3     y_4     y_5    │   y_6    │
  └─────────────────────────┴──────────┘

```

#### Key Temporal Feature Categories:

1. **Lag Features ($y_{t-k}$):** Historical targets shifted by $k$ temporal steps:

$$f_{\text{lag\_k}}(t) = y_{t-k}$$

2. **Rolling Window Statistics:** Aggregate metrics calculated over a fixed backward window of size $W$:

$$\mu_{\text{roll}}(t) = \frac{1}{W} \sum_{i=0}^{W-1} y_{t-i}, \quad \sigma_{\text{roll}}^2(t) = \frac{1}{W-1} \sum_{i=0}^{W-1} (y_{t-i} - \mu_{\text{roll}}(t))^2$$

3. **Expanding Window Statistics:** Cumulative aggregations computed over all available historical samples up to $t-1$:

$$\mu_{\text{exp}}(t) = \frac{1}{t-1} \sum_{i=1}^{t-1} y_i$$

4. **Cyclical Trigonometric Encodings:** Transforms periodic calendar features (e.g., hour $h \in [0, 23]$, month $m \in [1, 12]$) into continuous 2D coordinates using sine/cosine functions to preserve cyclic continuity:

$$x_{\text{sin}} = \sin\left(\frac{2\pi \cdot t}{P}\right), \quad x_{\text{cos}} = \cos\left(\frac{2\pi \cdot t}{P}\right)$$

Where $P$ is the fundamental period length (e.g., $P=24$ for hourly data).

---

### 5.5 Deep Learning Sequential Architectures

#### 1. Long Short-Term Memory (LSTM) Networks:

LSTMs address the **vanishing gradient problem** in standard RNNs using a persistent cell state $\mathbf{c}_t$ governed by three gating mechanisms:

$$\text{Forget Gate: } \mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

$$\text{Input Gate: } \mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

$$\text{Candidate Cell State: } \tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

$$\text{Cell State Update: } \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

$$\text{Output Gate: } \mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

$$\text{Hidden State: } \mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Where $\sigma(\cdot)$ is the sigmoid function and $\odot$ represents the Hadamard (element-wise) product.

```text
                        LSTM Cell Architectural Flow
                        
                        Cell State Path (c_{t-1} → c_t)
   c_{t-1} ─────────────( x )──────────────────(+)──────────────► c_t
                          ▲                     ▲
                          │ Forget Gate         │ Input Gate
                       ┌──┴──┐               ┌──┴──┐
                       │  f  │               │  i  │ * c~_t
                       └──▲──┘               └──▲──┘
                          │                     │
   h_{t-1} ──┬────────────┴─────────────────────┴───────┬────────► h_t
             │                                          │
             │   Output Gate (o_t) → [ o ] ──────────( x )
             │                         ▲               ▲
   x_t ──────┴─────────────────────────┴───────────────┴─ tanh(c_t)

```

#### 2. Gated Recurrent Units (GRU):

Simplifies the LSTM architecture by combining the cell state and hidden state into a single vector $\mathbf{h}_t$, using two gates:

* **Reset Gate:** $\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$
* **Update Gate:** $\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b})$$

#### 3. Temporal Convolutional Networks (TCN):

Uses 1D Dilated Causal Convolutions to process sequence data in parallel without recurrence.

* **Causal Convolutions:** Ensures predictions at step $t$ depend only on inputs at or before $t$, preventing future information leakage.
* **Dilated Convolutions:** Expands receptive field exponentially over deeper layers using dilation factor $d$:

$$y(t) = (x *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot x(t - d \cdot i)$$

---

### 5.6 Temporal Cross-Validation & Out-of-Sample Metrics

#### Leakage-Free Temporal Validation Schemes:

Standard $K$-Fold Cross-Validation shuffles sample indexes, causing severe **data leakage** in time-series data. Time-series cross-validation requires preserving temporal ordering.

1. **Expanding Window Cross-Validation:** The training set expands over time while the validation window slides forward.
2. **Rolling Window Cross-Validation:** Keeps the training window size fixed while sliding both the training and validation windows forward.

```text
                 Temporal Cross-Validation Schemes
                 
   Expanding Window Split:
   Fold 1: [ Train █ █ █ ] [ Val ░ ]
   Fold 2: [ Train █ █ █ █ █ ] [ Val ░ ]
   Fold 3: [ Train █ █ █ █ █ █ █ ] [ Val ░ ]
   
   Rolling Window Split:
   Fold 1: [ Train █ █ █ ] [ Val ░ ]
   Fold 2:   [ Train █ █ █ ] [ Val ░ ]
   Fold 3:     [ Train █ █ █ ] [ Val ░ ]

```

#### Forecasting Evaluation Metrics:

1. **Root Mean Squared Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N (y_t - \hat{y}_t)^2}$$

2. **Mean Absolute Scaled Error (MASE):** Independent of data scale; compares model error against a naïve baseline forecast ($y_t = y_{t-1}$):

$$\text{MASE} = \frac{\frac{1}{N} \sum_{t=1}^N \vert{}y_t - \hat{y}_t\vert{}}{\frac{1}{T-1} \sum_{i=2}^T \vert{}y_i - y_{i-1}\vert{}}$$

3. **Symmetric Mean Absolute Percentage Error (SMAPE):** Bounds percentage errors between $0\%$ and $200\%$:

$$\text{SMAPE} = \frac{100\%}{N} \sum_{t=1}^N \frac{\vert{}y_t - \hat{y}_t\vert{}}{(\vert{}y_t\vert{} + \vert{}\hat{y}_t\vert{}) / 2}$$

---

## 6. Enterprise Time-Series Forecasting Architecture

Production time-series systems feature modular pipelines for ingestion, stationarity checks, model selection, ensemble blending, and drift monitoring.

```text
             Enterprise Sequential Inference & Training Pipeline
┌─────────────────────────────────────────────────────────────────────────────┐
│ INGESTION & TEMPORAL PREPROCESSING ENGINE                                   │
│ • Resampling, Missing Value Imputation (Linear / Spline Interpolation)      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AUTOMATED STATIONARITY & DECOMPOSITION AUDIT                                │
│ • ADF & KPSS Tests → Apply Differencing (Δ^d) & Log Transformations         │
│ • Extract Trend & Seasonal Indices                                          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DUAL-PATH MODELING & HYPERPARAMETER SEARCH ENGINE                           │
│ ┌───────────────────────────┐         ┌───────────────────────────────────┐ │
│ │ Statistical Path (SARIMA) │         │ ML / DL Path (LGBM / LSTM / TCN) │ │
│ │ • Auto-ARIMA (AIC / BIC)  │         │ • Lag / Rolling Feature Engine    │ │
│ └─────────────┬─────────────┘         └─────────────────┬─────────────────┘ │
│               └───────────────────┬─────────────────────┘                   │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPANDING WINDOW TEMPORAL CROSS-VALIDATION & BLENDING ENSEMBLE              │
│ • Inverse Variance Weighting over Out-of-Sample MASE / RMSE                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PRODUCTION BACKTESTING, CONSTRAINTS & SERVING EXPORT                        │
│ • Out-of-sample Residual Drift Checks & MLflow Model Registry               │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Metric & Selection Matrix

| Metric | Domain / Target | Value Range | Optimal Target | Primary Characteristics & Best Use Case |
| --- | --- | --- | --- | --- |
| **RMSE** | Continuous | $[0, \infty)$ | $0.0$ | Heavily penalizes large forecast errors. Useful when large unexpected errors are costly. |
| **MAE** | Continuous | $[0, \infty)$ | $0.0$ | Measures average error magnitude. Robust to occasional unexpected spikes. |
| **MAPE** | Continuous Percentage | $[0, \infty)$ | $0.0\%$ | Measures percentage error relative to target. **Fails when actual values $y_t = 0$**. |
| **SMAPE** | Bounded Percentage | $[0\%, 200\%]$ | $0.0\%$ | Symmetric percentage error metric bounded between $0\%$ and $200\%$. Robust for scale-free comparisons. |
| **MASE** | Scale-Independent Ratio | $[0, \infty)$ | $< 1.0$ | **Industry standard metric for comparing forecasting models across series with different scales**. Values $< 1.0$ indicate performance better than a naïve baseline forecast. |

---

## 8. Technology & Implementation Matrix

| Framework / Library | Primary Modules | Optimization Focus | Recommended Production Use Case |
| --- | --- | --- | --- |
| **Statsmodels** | `tsa.arima.model.ARIMA`, `tsa.statespace.sarimax.SARIMAX` | Maximum Likelihood Estimation (MLE) via Kalman Filtering | Classical univariate time-series modeling, stationarity testing, and econometric forecasting. |
| **Prophet (Meta)** | `Prophet()` | Additive Generalized Additive Models (GAMs) with Bayesian Priors | Business forecasting with strong seasonal patterns, structural trend breaks, and holiday effects. |
| **LightGBM / XGBoost** | `LGBMRegressor`, `XGBRegressor` | Gradient Boosting over Lag and Rolling Features | High-throughput, multi-series tabular forecasting pipelines with exogenous features. |
| **PyTorch / Darts** | `nn.LSTM`, `Darts.models.TCNModel`, `NBEATSModel` | Sequence-to-Sequence Deep Learning Loss Optimization | Complex, large-scale, multivariate temporal forecasting with non-linear dependencies. |

---

## 9. Personal Understanding

Task 01 provides a comprehensive overview of time-series analysis and forecasting.

Key personal insights include:

1. **Stationarity is fundamental for statistical forecasting:** Models like ARIMA rely on constant statistical properties (mean, variance, autocovariance) over time. Testing for unit roots using **ADF** and **KPSS** tests before modeling prevents spurious regressions.
2. **Preventing temporal data leakage is critical:** Standard random $K$-Fold cross-validation leaks future information into historical training sets, yielding unrealistically optimistic performance metrics. Using **Expanding** or **Rolling Window Cross-Validation** preserves temporal sequence integrity and provides accurate out-of-sample error estimates.
3. **Machine learning feature engineering outperforms raw sequence inputs on tabular data:** Algorithms like LightGBM and XGBoost cannot inherently interpret time order. Explicitly engineering **lags**, **rolling window statistics**, and **cyclical trigonometric encodings** enables tree ensembles to match or beat deep sequential models on structured time-series tasks.

The foundational principle remains:

> **Time-series modeling exploits temporal autocorrelation, structural trends, and recurring seasonal patterns to construct mapping functions from historical states to future horizons while maintaining strict temporal ordering to prevent data leakage.**

---

## 10. Interview / Viva Questions

### Q1. Explain the difference between Strict Stationarity and Weak (Second-Order) Stationarity.

**Answer:**

* **Strict Stationarity:** Requires the joint probability distribution of any subset of observations $y_{t_1}, y_{t_2}, \dots, y_{t_k}$ to be identical to the shifted subset $y_{t_1+\tau}, y_{t_2+\tau}, \dots, y_{t_k+\tau}$ for any temporal shift $\tau$.
* **Weak Stationarity:** Requires only that the first two statistical moments are time-invariant:
1. Mean $\mathbb{E}[y_t] = \mu$ is constant $\forall t$.
2. Variance $\text{Var}(y_t) = \sigma^2 < \infty$ is constant $\forall t$.
3. Autocovariance $\text{Cov}(y_t, y_{t-k}) = \gamma(k)$ depends only on lag length $k$, not absolute time $t$.



### Q2. How do the ADF and KPSS tests differ, and how should they be used together?

**Answer:**

* **ADF Test:** Null Hypothesis $H_0$: The series has a **Unit Root** (Non-Stationary). $H_1$: Stationary.
* **KPSS Test:** Null Hypothesis $H_0$: The series is **Trend-Stationary**. $H_1$: Non-Stationary.
Using both tests provides a robust assessment of stationarity:
* **ADF Reject ($p < 0.05$) & KPSS Fail to Reject ($p > 0.05$):** The series is Stationary.
* **ADF Fail to Reject ($p > 0.05$) & KPSS Reject ($p < 0.05$):** The series is Non-Stationary (requires differencing).

### Q3. Derive the Backshift Operator expression for an $\text{ARIMA}(1, 1, 1)$ model.

**Answer:**

An $\text{ARIMA}(1, 1, 1)$ model combines an $\text{AR}(1)$ process, single differencing ($d=1$), and an $\text{MA}(1)$ process:

$$(1 - \phi_1 B) \Delta y_t = (1 + \theta_1 B) \epsilon_t$$

Since $\Delta y_t = (1 - B) y_t$:

$$(1 - \phi_1 B)(1 - B) y_t = (1 + \theta_1 B) \epsilon_t$$

Expanding the left-hand operator product:

$$\left(1 - (1 + \phi_1) B + \phi_1 B^2\right) y_t = (1 + \theta_1 B) \epsilon_t$$

Expressing without backshift operators:

$$y_t - (1 + \phi_1) y_{t-1} + \phi_1 y_{t-2} = \epsilon_t + \theta_1 \epsilon_{t-1}$$

$$y_t = (1 + \phi_1) y_{t-1} - \phi_1 y_{t-2} + \epsilon_t + \theta_1 \epsilon_{t-1}$$

### Q4. How do ACF and PACF plots help determine the orders $(p, q)$ of an $\text{ARMA}(p, q)$ process?

**Answer:**

* **$\text{AR}(p)$ Process:** The ACF tails off exponentially or sinusoidally, while the PACF cuts off sharply after lag $p$.
* **$\text{MA}(q)$ Process:** The ACF cuts off sharply after lag $q$, while the PACF tails off exponentially.
* **$\text{ARMA}(p, q)$ Process:** Both ACF and PACF tail off exponentially without sharp cutoffs.

```text
               ACF / PACF Identification Summary
               
        Model     ACF Pattern                 PACF Pattern
        ───────   ─────────────────────────   ─────────────────────────
        AR(p)     Tails off exponentially     Cuts off after lag p
        MA(q)     Cuts off after lag q        Tails off exponentially
        ARMA(p,q) Tails off exponentially     Tails off exponentially

```

### Q5. Why is standard $K$-Fold Cross-Validation inappropriate for time-series data, and what should be used instead?

**Answer:**

Standard $K$-Fold CV randomly shuffles observations, which breaks temporal dependencies and uses future observations to predict past states (**Data Leakage**).

Instead, **Time-Series Cross-Validation** (such as **Expanding Window** or **Rolling Window** CV) should be used. These methods maintain temporal ordering by ensuring training data always precedes validation data.

### Q6. What is the vanishing gradient problem in Recurrent Neural Networks (RNNs), and how do LSTMs resolve it?

**Answer:**

In standard RNNs, backpropagating gradients through long time sequences causes gradients to decay exponentially toward zero due to repeated matrix multiplications with weight matrices $\mathbf{W}_{hh} < 1$.

LSTMs resolve this using a **Cell State ($\mathbf{c}_t$)** that acts as an additive memory highway. Gating mechanisms (**Forget Gate $\mathbf{f}_t$**, **Input Gate $\mathbf{i}_t$**) regulate information flow, allowing gradients to flow back through time uninterrupted without exponential decay.

### Q7. How do Gated Recurrent Units (GRUs) differ from Long Short-Term Memory (LSTM) networks?

**Answer:**

1. **Gates:** LSTMs use three gates (Forget, Input, Output), whereas GRUs simplify this to two gates (**Reset Gate $\mathbf{r}_t$** and **Update Gate $\mathbf{z}_t$**).
2. **State Vectors:** LSTMs maintain separate cell ($\mathbf{c}_t$) and hidden ($\mathbf{h}_t$) states. GRUs merge them into a single hidden state ($\mathbf{h}_t$).
3. **Efficiency:** GRUs have fewer parameters, making them faster to train and less prone to overfitting on smaller datasets.

### Q8. What is a Temporal Convolutional Network (TCN), and why is it often preferred over LSTMs for sequence modeling?

**Answer:**

A TCN uses **1D Dilated Causal Convolutions** combined with residual connections to process temporal sequences.

Advantages over LSTMs:

1. **Parallelization:** Convolutions process the entire temporal sequence in parallel during training, avoiding the sequential bottleneck of RNNs.
2. **No Vanishing Gradients:** The architecture maintains stable gradients across long temporal sequences.
3. **Flexible Receptive Field:** Increasing the dilation factor $d$ exponentially expands the model's historical context window.

### Q9. Why must cyclical features (such as hour of day or day of week) be transformed into trigonometric sine and cosine features?

**Answer:**

Integer representations of cyclical features create an artificial jump at boundaries (e.g., hour $23$ to hour $0$). Linear and tree-based models interpret this continuous transition as a maximum numerical jump ($\Delta = 23$).

Mapping cycles onto continuous 2D Cartesian coordinates using sine and cosine functions preserves true distance relationships:

$$x_{\text{sin}} = \sin\left(\frac{2\pi \cdot t}{P}\right), \quad x_{\text{cos}} = \cos\left(\frac{2\pi \cdot t}{P}\right)$$

This ensures that $t=23$ and $t=0$ are close in feature space.

```text
               Cyclical Trigonometric Projection (P=24)
                           x_cos (1.0)
                               │  t=0 (Midnight)
                        ┌──────┴──────┐
                        │             │
        x_sin (-1.0) ───┤   Circle    ├─── x_sin (+1.0)
         t=18 (6 PM)    │             │    t=6 (6 AM)
                        └──────┬──────┘
                               │  t=12 (Noon)
                          x_cos (-1.0)

```

### Q10. What is Mean Absolute Scaled Error (MASE), and why is it preferred over MAPE for time-series evaluation?

**Answer:**

* **MAPE Problem:** Breaks down or produces infinite values when actual targets $y_t = 0$, and heavily penalizes positive errors more than negative errors.
* **MASE Solution:** Scales absolute forecast errors relative to the mean absolute error of an in-sample naïve baseline forecast ($y_t = y_{t-1}$):

$$\text{MASE} = \frac{\frac{1}{N} \sum_{t=1}^N \vert{}y_t - \hat{y}_t\vert{}}{\frac{1}{T-1} \sum_{i=2}^T \vert{}y_i - y_{i-1}\vert{}}$$

MASE is scale-independent, well-defined when $y_t = 0$, and provides a direct baseline benchmark: $\text{MASE} < 1.0$ indicates the model outperforms a naïve forecast.

### Q11. Explain the difference between Additive and Multiplicative classical structural decomposition.

**Answer:**

* **Additive Model ($y_t = T_t + S_t + I_t$):** Used when the amplitude of seasonal variations remains constant over time regardless of changes in the underlying trend level.
* **Multiplicative Model ($y_t = T_t \times S_t \times I_t$):** Used when seasonal variations scale proportionally with changes in the trend level (e.g., seasonal peaks grow larger as the overall trend increases).

### Q12. What is exponential smoothing, and how does Holt-Winters extend simple exponential smoothing?

**Answer:**

* **Simple Exponential Smoothing (SES):** Forecasts using exponentially decreasing weights on historical observations. Suitable for stationary series without trend or seasonality.
* **Holt's Linear Method:** Extends SES by adding a separate equation to model linear **Trend**.
* **Holt-Winters Method (Triple Exponential Smoothing):** Extends Holt's method by adding a third equation to model **Seasonality** (additive or multiplicative), enabling forecasting for series with both trends and seasonal patterns.

### Q13. How do exogenous variables ($X_t$) integrate into a $\text{SARIMAX}$ model?

**Answer:**

$\text{SARIMAX}$ incorporates exogenous input features $X_t$ by modeling target $y_t$ as a linear combination of exogenous features plus a $\text{SARIMA}$ residual error process:

$$y_t = \sum_{k=1}^m \beta_k X_{k, t} + \eta_t$$

$$\Phi_p(B) \tilde{\Phi}_P(B^s) (1 - B)^d (1 - B^s)^D \eta_t = \Theta_q(B) \tilde{\Theta}_Q(B^s) \epsilon_t$$

This allows the model to capture external drivers (such as promotions or weather conditions) alongside underlying temporal dynamics.

### Q14. What is systematic data leakage in time-series feature engineering, and how can it be avoided?

**Answer:**

Data leakage occurs when target information from the future ($t + k$) is accidentally included in features used to train past or current steps ($t$).

Examples include:

* Computing rolling statistics centered on time $t$ using future values ($t+1, \dots, t+k$).
* Applying global scalers or imputers fit across the entire dataset instead of using fit parameters calculated strictly on historical training folds.
**Prevention:** Calculate all rolling windows, scalers, and imputers using strictly backward-looking windows ($t-k \dots t-1$).

### Q15. Compare the performance characteristics of LightGBM with lag features versus an LSTM network for high-cardinality multi-series forecasting.

**Answer:**

* **LightGBM (Tabular Approach):** Extremely fast to train, highly scalable across thousands of parallel series using categorical embeddings, handles tabular exogenous metadata easily, and requires less tuning. However, it cannot inherently model long temporal sequences without explicit feature engineering.
* **LSTM (Sequential Approach):** Automatically learns complex, non-linear long-term temporal dependencies without requiring explicit feature engineering. However, it is computationally expensive, sensitive to hyperparameter choices, and prone to overfitting on smaller or noisy tabular datasets.

---

## 11. Conclusion

Task 01 covers the foundational principles of Time-Series Analysis, Stochastic Modeling, Feature Engineering, and Deep Sequential Architectures.

```text
               Time-Series Modeling Execution Pathway
                                  ↓
Data Ingestion, Frequency Regularization, & Spline/Linear Interpolation
                                  ↓
Classical Structural Decomposition & Stationarity Auditing (ADF & KPSS Tests)
                                  ↓
Stationarity Transformations (Differencing Δ^d, Log / Box-Cox Variance Stabilization)
                                  ↓
Feature Construction (Lags, Rolling Stats, Cyclical Sine/Cosine Encodings)
                                  ↓
Model Fitting (SARIMAX Statistical Baseline vs LightGBM / LSTM / TCN Architectures)
                                  ↓
Temporal Expanding Window Cross-Validation & Metric Evaluation (MASE, SMAPE, RMSE)

```

The core structural pillars of Time-Series Analysis include:

```text
Time-Series Analysis & Forecasting Pillars
├── Structural Decomposition & Stationarity Dynamics (Trend, Seasonality, ADF/KPSS, Differencing)
├── Stochastic Autoregressive Models (AR, MA, ARMA, ARIMA, SARIMA Order Rules via ACF/PACF)
├── Supervised Temporal Feature Engineering (Lags, Rolling Aggregations, Cyclical Encodings)
└── Deep Sequential Models & Temporal Validation (LSTM, GRU, TCN, Expanding Window CV, MASE)

```

Core tools and operational frameworks:

```text
Statsmodels Framework (Statistical time-series analysis, SARIMAX, ADF/KPSS testing)
LightGBM & XGBoost (High-throughput GBDT forecasting over engineered temporal features)
PyTorch Deep Learning Engine (Sequential models: LSTM, GRU, Temporal Convolutional Networks)
Darts & Prophet Libraries (Unified APIs for classical, machine learning, and neural forecasting)

```

Completing Task 01 provides the theoretical understanding and practical framework needed to build production forecasting pipelines, prevent data leakage, evaluate temporal models accurately, and deploy high-performance sequential architectures.

The foundational principle remains:

> **Time-series modeling exploits temporal autocorrelation, structural trends, and recurring seasonal patterns to construct mapping functions from historical states to future horizons while maintaining strict temporal ordering to prevent data leakage.**

---

## 12. Key Takeaways

1. **Time-series data** consists of sequentially ordered observations where temporal dependence (autocorrelation) must be modeled explicitly.
2. **Classical Decomposition** breaks down a series into **Trend ($T_t$)**, **Seasonality ($S_t$)**, and **Irregular Noise ($I_t$)** using additive or multiplicative models.
3. **Weak Stationarity** requires a constant mean, constant variance, and autocovariance that depends only on lag length $k$.
4. **ADF and KPSS tests** assess stationarity; non-stationary series require differencing ($\Delta^d$) to remove unit roots.
5. **Autoregressive ($\text{AR}(p)$)** models capture dependencies on past values, while **Moving Average ($\text{MA}(q)$)** models capture dependencies on past forecast errors.
6. **ACF and PACF plots** help identify model orders: PACF cutoffs indicate $\text{AR}(p)$ order, while ACF cutoffs indicate $\text{MA}(q)$ order.
7. **$\text{SARIMA}(p, d, q) \times (P, D, Q)_s$** extends ARIMA to model both non-seasonal and seasonal trends with period length $s$.
8. **Supervised ML models (LightGBM, XGBoost)** require transforming raw series into tabular format using **lags**, **rolling statistics**, and **cyclical trigonometric encodings**.
9. **LSTMs and GRUs** resolve the vanishing gradient problem in standard RNNs using gating mechanisms, making them suitable for learning long-term temporal dependencies.
10. **Temporal Convolutional Networks (TCNs)** use 1D dilated causal convolutions to process sequence data in parallel without recurrence.
11. **Time-Series Cross-Validation** (Expanding or Rolling Window) must preserve temporal ordering to prevent data leakage.
12. **MASE (Mean Absolute Scaled Error)** is the standard scale-independent metric for evaluating forecasting models against a naïve baseline.
