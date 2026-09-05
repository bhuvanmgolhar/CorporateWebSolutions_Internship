# Task 02 — The Derivation of Rules from Data Analysis: Association Rule Mining, Inductive Rule Learning & Decision Tree Rule Extraction

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal VI |
| Task Number | 02 |
| Topic | Rule Derivation from Data Analysis, Association Rule Mining, Frequent Itemset Mining, Apriori & FP-Growth, Inductive Rule Learning, Decision Tree Rule Extraction |
| Task Type | Applied Data Mining & Inductive Statistical Learning |
| Status | Completed |
| Repository Section | `tasks/portal-06/task-02/` |

---

## 2. Objective

The objective of this task is to study, implement, and mathematically analyze **Methods for Deriving Rules from Empirical Data Analysis**. 
This task focuses on:
- Transitioning from human-enumerated symbolic logic (Task 01) to empirical rule discovery ($\text{Data} + \text{Labels/Transactions} \to \text{Discovered Rules}$).
- Formalizing unsupervised rule mining via **Frequent Itemset Mining** and **Association Rule Mining** (Market Basket Analysis).
- Evaluating algorithmic efficiency between candidate-generation approaches (Apriori) and tree-based suffix pattern mining (FP-Growth).
- Deriving explicit, interpretable $\text{IF-THEN}$ rule sets from supervised machine learning models like Decision Trees, Decision Lists, and RuleFit.
- Analyzing statistical significance metrics: Support, Confidence, Lift, Conviction, and Leverage to filter out spurious correlation rules.

---

## 3. Introduction

While Task 01 addressed the manual declaration of explicit rules by domain experts, **Task 02 explores how actionable IF-THEN rules are automatically extracted from unstructured or tabular data through empirical analysis**. 

```text
               Deductive Rule Enumeration vs. Inductive Rule Derivation
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEDUCTIVE PARADIGM (Task 01: Explicit Rule Enumeration)                     │
│ Domain Knowledge  ───► Hand-Crafted IF-THEN Rules ───► Evaluated on Data    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ INDUCTIVE PARADIGM (Task 02: Empirical Rule Derivation)                     │
│ Transactional/Tabular Data ───► Mining / Induction Algorithm ───► Rules      │
└─────────────────────────────────────────────────────────────────────────────┘

```

Automated rule derivation solves the combinatorial bottleneck of manual logic creation. By applying frequent pattern mining algorithms or extracting decision paths from non-parametric decision trees, data scientists can convert massive datasets into human-understandable, deterministic logic rules.

The core principle governing rule derivation from data is:

> **Automated rule derivation bridges the gap between unstructured big data and interpretable logic by extracting statistically significant, minimal-entropy conditional rules without manual logic engineering.**

---

## 4. Paradigm Comparison Matrix

Understanding rule derivation methods requires comparing unsupervised association rule algorithms with supervised inductive rule learners.

```text
               Rule Derivation Paradigms Across Mining Tasks
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Algorithmic Paradigm                  │ Primary Data Processing Strategy      │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Unsupervised Pattern Mining           │ Scans transactional databases to find │
│ (Apriori, FP-Growth, ECLAT)           │ co-occurring items and conditional    │
│                                       │ probabilities without target labels. │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Tree-Based Rule Extraction            │ Splits feature space via recursive    │
│ (Decision Trees, CART, C4.5)          │ entropy minimization; extracts paths  │
│                                       │ as Disjunctive Normal Form (DNF).     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Sequential & Inductive Rule Learners  │ Generates rule sets directly using    │
│ (RIPPER, CN2, RuleFit)                │ separate-and-conquer strategy or     │
│                                       │ sparse linear regression over trees.  │
└───────────────────────────────────────┴───────────────────────────────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

Formalizing rule derivation requires analyzing transaction probability metrics, candidate pruning strategies, and information theory.

### 5.1 Association Rule Mining Metrics

Let $I = \{i_1, i_2, \dots, i_m\}$ be a set of binary literals called *items*, and let $T = \{t_1, t_2, \dots, t_n\}$ be a set of transactions, where each transaction $t_k \subseteq I$. An association rule is an implication of the form:

$$X \Rightarrow Y \quad \text{where } X, Y \subseteq I \text{ and } X \cap Y = \emptyset$$

To evaluate the strength and validity of derived rule $X \Rightarrow Y$, four foundational mathematical metrics are calculated:

#### 1. Support

Measures the proportion of transactions in the database that contain both itemsets $X$ and $Y$:

$$\text{Supp}(X \Rightarrow Y) = P(X \cap Y) = \frac{\vert{}\{t \in T \mid X \cup Y \subseteq t\}\vert{}}{\vert{}T\vert{}}$$

#### 2. Confidence

Measures the conditional probability that a transaction contains $Y$, given that it contains $X$:

$$\text{Conf}(X \Rightarrow Y) = P(Y \mid X) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X)}$$

#### 3. Lift

Measures how much more often $X$ and $Y$ occur together than would be expected if they were statistically independent:

$$\text{Lift}(X \Rightarrow Y) = \frac{P(X \cap Y)}{P(X) \cdot P(Y)} = \frac{\text{Conf}(X \Rightarrow Y)}{\text{Supp}(Y)}$$

* $\text{Lift} = 1$: $X$ and $Y$ are completely independent.
* $\text{Lift} > 1$: Positive correlation ($X$ boosts the occurrence of $Y$).
* $\text{Lift} < 1$: Negative correlation / substitution effect.

#### 4. Conviction

Measures the ratio of the expected frequency that $X$ occurs without $Y$ if they were independent, divided by the observed frequency of incorrect predictions:

$$\text{Conv}(X \Rightarrow Y) = \frac{1 - \text{Supp}(Y)}{1 - \text{Conf}(X \Rightarrow Y)} = \frac{P(X) \cdot P(\neg Y)}{P(X \cap \neg Y)}$$

---

### 5.2 Algorithmic Implementations: Apriori vs. FP-Growth

#### The Apriori Principle (Anti-Monotonicity Property)

The computational bottleneck of naive rule extraction is $2^{\vert{}I\vert{}}$ itemset evaluation. Apriori avoids this using the **Downward Closure Property**:

$$\text{If an itemset } I \text{ is infrequent, all of its supersets } I' \supset I \text{ must also be infrequent.}$$

$$\text{Supp}(I') \le \text{Supp}(I) < \text{min\_supp}$$

```text
                     Apriori Search Space Pruning
                          [A, B] (Infrequent)
                        /        \
                   [A, B, C]    [A, B, D]    (Pruned dynamically)
                   /       \    /       \
               [A,B,C,D] ...  ...       ...  (Never calculated)

```

#### The FP-Growth Algorithm (Frequent Pattern Tree)

While Apriori requires $k$ database passes for itemsets of length $k$, **FP-Growth** uses a compact memory structure called the **FP-Tree** requiring only **two database scans**:

1. **Scan 1:** Count individual item frequencies and sort items in descending order of frequency.
2. **Scan 2:** Construct the FP-Tree by mapping transactions directly onto shared tree branches.
3. **Recursive Mining:** Perform pattern fragment growth using Conditional FP-Trees without candidate generation.

```text
                   FP-Tree Compact Memory Representation
                                 (Root)
                                /      \
                           {Milk: 4}  {Bread: 2}
                            /     \
                     {Diaper: 3}  {Diaper: 1}
                       /
                  {Beer: 3}

```

---

### 5.3 Rule Extraction from Supervised Decision Trees

In supervised learning, explicit conditional rules are derived from trained Decision Trees (e.g., CART or C4.5). Splitting criteria rely on **Information Gain** or **Gini Impurity**:

$$\text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2$$

$$\text{Entropy}(D) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

Every path from the root node to a leaf node represents a distinct antecedent-consequent rule:

$$R_k: \bigwedge_{j \in \text{Path}_k} \left( x_j \odot v_j \right) \implies y = C_k$$

Where $\odot \in \{\le, >, =, \in\}$ represents feature logic splits.

```text
                     Decision Tree Path Rule Derivation
                                [Income > $50k?]
                                 /            \
                               Yes             No
                              /                 \
                     [Credit < 650?]         ===> Rule 1: IF Income <= 50k THEN Deny
                      /           \
                    Yes            No
                    /               \
            ===> Rule 2:        ===> Rule 3:
           IF Income > 50k     IF Income > 50k
           AND Credit < 650    AND Credit >= 650
           THEN Deny           THEN Approve

```

---

## 6. Enterprise Rule Derivation System Architecture

A production pipeline for deriving, filtering, and deploying rules from empirical data follows a structured end-to-end data processing lifecycle:

```text
                     Empirical Rule Derivation Pipeline
┌─────────────────────────────────────────────────────────────────────────────┐
│ RAW DATA SOURCES (Transactions, Customer Logs, Feature Tables)              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA PREPROCESSING & DISCRETIZATION                                         │
│ (Binned continuous features, Transaction Encoding, One-Hot Transformation)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ RULE DERIVATION ENGINE                                                      │
│ ┌───────────────────────────┐         ┌───────────────────────────────────┐ │
│ │ Unsupervised Pattern Mine │   OR    │ Supervised Tree / Sequential Mine │ │
│ │ (Apriori / FP-Growth)     │         │ (CART / C4.5 / RIPPER Engine)     │ │
│ └───────────────────────────┘         └───────────────────────────────────┘ │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ METRIC FILTERING & PRUNING (Filter by Supp >= min_supp, Lift > 1.2)         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXECUTABLE RULE SET / PRODUCTION KNOWLEDGE BASE                             │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 7. Comparative Analysis & Decision Framework

Selecting the optimal rule derivation methodology depends on whether target labels exist and the dimensionality of the feature space.

| Metric / Feature | Apriori Mining | FP-Growth Mining | Decision Tree Extraction | Sequential Rule Mining (RIPPER) |
| --- | --- | --- | --- | --- |
| **Target Label** | Unsupervised (None) | Unsupervised (None) | Supervised (Class target) | Supervised (Class target) |
| **Database Scans** | $k$ passes ($k$=max itemset) | Exactly 2 scans | Single pass over in-memory dataset | Iterative passes per class |
| **Memory Consumption** | High (candidate generation) | Low (compact FP-Tree) | Medium (tree structure) | Low (greedy separate & conquer) |
| **Execution Speed** | Slow on large databases | Extremely fast | Fast | Moderate |
| **Continuous Data** | Requires discretization | Requires discretization | Native splitting handling | Native splitting handling |
| **Rule Format** | $X \Rightarrow Y$ (Itemsets) | $X \Rightarrow Y$ (Itemsets) | DNF Paths ($\text{IF } A \wedge B \to C$) | Ordered Decision List |

---

## 8. Technology & Integration Matrix

| Functional Role | Industry Standard Tooling | Primary Operational Function |
| --- | --- | --- |
| **Association Mining Libraries** | `mlxtend` (Python), `arules` (R), PySpark MLlib | Performs Apriori, FP-Growth, and ECLAT frequent itemset extraction. |
| **Tree & Rule Extraction** | `scikit-learn` (`DecisionTreeClassifier`), `RuleFit`, `wittgenstein` | Derives decision trees and extracts minimal RIPPER/RuleFit rule sets. |
| **Big Data Engine** | Apache Spark (FP-Growth distributed implementation) | Scales frequent pattern derivation across billions of transactional logs. |
| **Rule Export Formats** | PMML (Predictive Model Markup Language), PFA | Serializes derived statistical rules into interoperable XML/JSON for rule engines. |

---

## 9. Personal Understanding

Task 02 completes the complementary perspective to Task 01.
While Task 01 required hand-crafting explicit rules using expert logic, Task 02 proves that **rules can be automatically derived from empirical data analysis**.
I now understand that association rule metrics like **Support**, **Confidence**, and **Lift** act as statistical filters, ensuring that only meaningful patterns are extracted. Algorithms like **FP-Growth** optimize runtime performance by replacing repeated dataset scans with compact tree structures. Furthermore, deriving rules from Decision Trees provides a bridge between complex machine learning models and interpretable business logic.
The core principle remains:

> **Automated rule derivation bridges the gap between unstructured big data and interpretable logic by extracting statistically significant, minimal-entropy conditional rules without manual logic engineering.**

---

## 10. Interview / Viva Questions

### Q1. What is the fundamental difference between explicit rule enumeration (Task 01) and derived rule analysis (Task 02)?

**Answer:**

Explicit rule enumeration involves human experts manually encoding known business policies into logic frameworks ($\text{Knowledge} \to \text{Rules}$). Derived rule analysis uses data mining and machine learning algorithms to discover unknown, statistically supported patterns directly from transactional or tabular data ($\text{Data} \to \text{Rules}$).

### Q2. How is Support calculated, and why is a minimum support threshold necessary?

**Answer:**

Support is the proportion of total transactions containing an itemset $X$: $\text{Supp}(X) = \frac{\vert{}\{t \in T \mid X \subseteq t\}\vert{}}{\vert{}T\vert{}}$. A minimum support threshold (`min_supp`) is required to filter out rare item combinations, significantly reducing the exponential search space and preventing overfitting to noise.

### Q3. Why is Confidence alone insufficient to determine the quality of an association rule?

**Answer:**

Confidence measures $P(Y \mid X)$, but ignores the overall baseline frequency of $Y$ ($P(Y)$). If $Y$ is an extremely frequent item (e.g., $P(Y) = 90\%$), a rule $X \Rightarrow Y$ might have a high confidence of $90\%$ even if $X$ has no actual predictive relation to $Y$. **Lift** is needed to account for $P(Y)$.

### Q4. What does a Lift value less than 1.0 signify in Market Basket Analysis?

**Answer:**

A Lift value less than 1.0 indicates a **negative correlation** or substitution effect between itemsets $X$ and $Y$. It means the occurrence of $X$ actually decreases the likelihood that $Y$ will occur in the same transaction.

### Q5. How does the Apriori algorithm use the Downward Closure Property to reduce computational time?

**Answer:**

The Downward Closure Property states that if an itemset $I$ is infrequent (below `min_supp`), then all of its supersets $I' \supset I$ must also be infrequent. Apriori uses this to prune entire branches of candidate itemsets without scanning the database to evaluate them.

### Q6. Why is FP-Growth significantly faster than the Apriori algorithm on large datasets?

**Answer:**

Apriori requires $k$ full database scans for candidate itemsets of length $k$ and generates massive candidate sets. FP-Growth compresses the database into an **FP-Tree** using only **two database scans**, completely eliminating explicit candidate generation by recursively mining conditional trees.

### Q7. How are explicit rules extracted from a trained Decision Tree model?

**Answer:**

Every distinct path from the root node of a decision tree to a leaf node represents a conditional rule. The internal node decisions form the antecedent ($\text{IF}$ conditions joined by $\text{AND}$), and the leaf node class prediction forms the consequent ($\text{THEN}$ class prediction).

### Q8. What is the concept of "Separate-and-Conquer" in sequential rule learning algorithms like RIPPER?

**Answer:**

Separate-and-Conquer learns one rule at a time that covers a subset of positive examples, removes (separates) the covered examples from the training set, and recursively learns the next rule on the remaining dataset until all positive instances are covered.

### Q9. What is the mathematical definition of Conviction, and how does it handle directional rule implications?

**Answer:**

$$\text{Conv}(X \Rightarrow Y) = \frac{1 - \text{Supp}(Y)}{1 - \text{Conf}(X \Rightarrow Y)}$$


Conviction measures the ratio of expected incorrect predictions assuming independence over actual incorrect predictions. Unlike Lift, Conviction is **directional** ($\text{Conv}(X \Rightarrow Y) \neq \text{Conv}(Y \Rightarrow X)$), evaluating logical implication strength.

### Q10. What is continuous feature discretization, and why is it necessary for association rule mining?

**Answer:**

Association rule algorithms (Apriori, FP-Growth) operate on discrete, categorical item tokens. Continuous features (e.g., `Age = 28.5`, `Income = 62400`) must be discretized into categorical intervals or bins (e.g., `Age_20_to_30`, `Income_Medium`) prior to itemset creation.

### Q11. How does the RuleFit algorithm combine tree-based rule extraction with linear modeling?

**Answer:**

RuleFit builds an ensemble of decision trees, extracts all decision paths as binary rule features, and then trains a sparse linear model (Lasso / $L_1$ regularization) across both original features and extracted rule features to select the most predictive, minimal rule set.

### Q12. What is the ECLAT algorithm, and how does its data layout differ from Apriori?

**Answer:**

ECLAT (Equivalence Class Transformation) uses a **vertical data format** ($\text{Item} \to \text{Transaction IDs}$) rather than a horizontal format ($\text{Transaction ID} \to \text{Items}$). It intersects transaction ID lists (TID-lists) to calculate itemset support directly without scanning horizontal rows.

### Q13. How do missing values affect derived association rules compared to decision trees?

**Answer:**

In association rule mining, missing values are simply omitted from itemsets (treated as non-occurrences). In decision trees, missing values require imputation or alternative split routing (surrogate splits) to preserve entropy calculations during split evaluation.

### Q14. What is PMML, and how is it used to deploy derived rules to enterprise systems?

**Answer:**

PMML (Predictive Model Markup Language) is an XML standard used to serialize trained statistical models and derived rule sets. A rule set derived in Python/R can be exported to PMML and loaded directly into a production enterprise Java/C++ rule engine.

### Q15. When should a data scientist use derived rule extraction instead of a deep neural network?

**Answer:**

Derived rule extraction should be used when:

1. Complete model interpretability and human auditability are required.
2. The domain requires explicit $\text{IF-THEN}$ logic for regulatory reporting.
3. The dataset consists of clear tabular/transactional features rather than unstructured spatial/temporal signals (e.g., image/audio).

---

## 11. Conclusion

Task 02 presents the methodology for deriving interpretable logical rules from data analysis.
The operational workflow for deriving rules from data is summarized below:

```text
Data Mining & Empirical Rule Extraction Lifecycle
      ↓
Raw Data Collection & Binned Discretization
      ↓
Frequent Pattern Mining (FP-Growth / Apriori) OR Tree Induction (CART)
      ↓
Rule Extraction (Path Aggregation / Itemset Implication)
      ↓
Statistical Pruning & Filtering (Support, Confidence, Lift, Conviction)
      ↓
Export to Interoperable Formats (PMML / Decision Tables)

```

The core structural pillars of derived rule engineering include:

```text
Derived Rule Analysis Framework
├── Unsupervised Pattern Mining (Apriori, FP-Growth, ECLAT)
├── Supervised Rule Extraction (Decision Tree Path Conversion, RIPPER)
├── Statistical Quality Metrics (Support, Confidence, Lift, Conviction)
└── Deployment Formats (PMML Serialization, Executable DMN Rules)

```

Core tools and operational frameworks:

```text
mlxtend / arules / PySpark MLlib
scikit-learn / RuleFit / wittgenstein
Apache Spark FP-Growth Engine
PMML Serializer / Decision Table Converter

```

By completing Task 02, data scientists master the ability to automatically discover, prune, and extract explicit $\text{IF-THEN}$ rules from complex transactional and tabular datasets.
The central principle remains:

> **Automated rule derivation bridges the gap between unstructured big data and interpretable logic by extracting statistically significant, minimal-entropy conditional rules without manual logic engineering.**

---

## 12. Key Takeaways

1. **Rule Derivation** automatically discovers deterministic $\text{IF-THEN}$ logic from transactional or tabular data ($\text{Data} \to \text{Rules}$).
2. **Association Rule Mining** identifies conditional relationships $X \Rightarrow Y$ in unsupervised transactional datasets.
3. **Support** measures the total frequency of an itemset in a database: $\text{Supp}(X) = P(X)$.
4. **Confidence** measures the conditional probability $P(Y \mid X) = \frac{\text{Supp}(X \cup Y)}{\text{Supp}(X)}$.
5. **Lift** evaluates correlation strength relative to independence: $\text{Lift}(X \Rightarrow Y) = \frac{\text{Conf}(X \Rightarrow Y)}{\text{Supp}(Y)}$.
6. **Conviction** provides a directional metric for logical implication strength based on expected error.
7. The **Apriori Algorithm** uses the Downward Closure Property to prune search paths of infrequent itemsets.
8. **FP-Growth** compresses databases into an **FP-Tree**, requiring only **two database scans** and zero candidate generation.
9. **Decision Trees** allow supervised rule derivation by converting root-to-leaf paths into Disjunctive Normal Form (DNF) rules.
10. **Information Gain** and **Gini Impurity** guide decision tree splitting to create optimal, low-entropy rule conditions.
11. **ECLAT** uses vertical transaction ID lists to compute support via set intersections.
12. **RIPPER** applies a separate-and-conquer strategy to build ordered decision lists directly from labeled data.
13. **Continuous Features** must be discretized into interval bins before association rule mining.
14. **PMML (Predictive Model Markup Language)** enables cross-platform deployment of derived rule sets to production engines.
15. Rule derivation combines the interpretability of symbolic AI with the automated discovery power of statistical learning.
