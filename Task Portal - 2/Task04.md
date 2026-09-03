# Task 04 — The CRISP-DM Model in Data Science

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal II |
| Task Number | 04 |
| Topic | The CRISP-DM Model in Data Science |
| Task Type | Conceptual / Methodology |
| Status | Completed |
| Repository Section | `tasks/portal-02/task-04/` |

---

## 2. Objective

The objective of this task is to understand the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** model and how it can be used to organize Data Science and Data Mining projects.

This task focuses on:

- Understanding what CRISP-DM means
- Learning the six major phases
- Understanding the purpose of each phase
- Understanding why the process is iterative
- Connecting CRISP-DM with real Data Science projects
- Learning the advantages and limitations of the model

---

## 3. Introduction

Data Science projects involve many different activities, from understanding a business problem to collecting data, building models, and delivering useful results.

Without a structured process, a project can easily become focused on technical work without solving the actual problem.

**CRISP-DM** provides a structured methodology for organizing Data Mining and Data Science projects.

CRISP-DM stands for:

> **Cross-Industry Standard Process for Data Mining**

The model is commonly represented using six major phases:

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment
        ↺
```

An important point is that this is **not a strict one-way process**. Teams may move backward and forward between phases as they learn more about the data and problem.

---

# 4. The Six Phases of CRISP-DM

The six phases are:

```text
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
```

Each phase has a different purpose.

---

# 5. Phase 1 — Business Understanding

## Definition

**Business Understanding** is the process of understanding the actual problem, objective, constraints, and expected outcome before beginning technical modeling.

This is often one of the most important phases because a technically successful model can still be useless if it does not address the real problem.

## Important Questions

A team may ask:

- What problem are we trying to solve?
- Why is the problem important?
- What does success mean?
- Who will use the result?
- What constraints exist?
- What business metric matters?

## Example

Suppose a telecom company wants to reduce customer churn.

The business objective might be:

```text
Reduce avoidable customer churn.
```

The Data Science objective could become:

```text
Predict customers who have a high probability of churning.
```

These are related, but they are not exactly the same.

---

# 6. Phase 2 — Data Understanding

Once the problem is defined, the next step is understanding the available data.

This phase can involve:

- Collecting initial data
- Examining data sources
- Understanding variables
- Exploring distributions
- Identifying missing values
- Detecting unusual observations
- Checking data quality

Example:

```text
Customer Data
      ↓
Age
Monthly Bill
Contract
Usage
Complaints
Churn
```

The team needs to understand what each variable means and whether the data is appropriate for the problem.

---

# 7. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is commonly performed during data understanding.

EDA can include:

- Summary statistics
- Histograms
- Bar charts
- Scatter plots
- Correlation analysis
- Group comparisons
- Outlier investigation

For example:

```text
Customer Churn
      ↓
Compare:
- Contract Type
- Monthly Bill
- Usage
- Complaints
```

EDA can help reveal patterns and potential data-quality problems before modeling.

---

# 8. Phase 3 — Data Preparation

**Data Preparation** involves transforming raw data into a dataset suitable for modeling or analysis.

This phase can include:

### Cleaning

- Handling missing values
- Removing duplicates
- Correcting invalid values

### Transformation

- Scaling numerical variables when appropriate
- Encoding categorical variables
- Converting data types

### Feature Engineering

Creating useful variables from existing information.

For example:

```text
Date of Purchase
      ↓
Month
Day of Week
Quarter
```

### Feature Selection

Choosing variables that are relevant to the task.

Data preparation can require substantial work in real-world projects.

---

# 9. Phase 4 — Modeling

The **Modeling** phase involves selecting and training suitable analytical or Machine Learning models.

The choice depends on:

- Problem type
- Data characteristics
- Performance requirements
- Interpretability requirements
- Available computational resources

Examples include:

```text
Regression
Classification
Clustering
Decision Trees
Random Forest
Gradient Boosting
Neural Networks
```

A simplified workflow is:

```text
Prepared Data
      ↓
Choose Technique
      ↓
Train Model
      ↓
Tune Parameters
      ↓
Generate Results
```

Different models may be tested and compared rather than selecting only one model immediately.

---

# 10. Model Training and Validation

During modeling, a dataset may be divided into appropriate subsets.

For supervised learning:

```text
Training Data
      ↓
Learn Model

Validation Data
      ↓
Tune / Compare Models

Test Data
      ↓
Final Unseen Evaluation
```

The exact strategy depends on the project and data.

The purpose is to avoid choosing a model simply because it performs well on the same data used to fit it.

---

# 11. Phase 5 — Evaluation

The **Evaluation** phase checks whether the model and overall results actually meet the project objectives.

This phase is broader than asking:

> "Is the model accurate?"

The team should also ask:

- Does the model solve the original problem?
- Are the results reliable?
- Are the evaluation metrics appropriate?
- Are there important limitations?
- Can the results support the intended decision?

## Example

A churn model may have strong classification metrics but still provide limited business value if the company cannot take any meaningful action based on the predictions.

Therefore:

```text
Technical Performance
        +
Business Relevance
        ↓
Useful Solution
```

---

# 12. Evaluation Metrics

The metric depends on the problem.

## Classification

Common metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Regression

Common metrics include:

- MAE
- MSE
- RMSE
- R²

The metric should reflect the actual costs and objectives of the problem.

---

# 13. Phase 6 — Deployment

**Deployment** means making the results available for practical use.

Deployment can take different forms.

Examples include:

- Dashboard
- Report
- Automated system
- API
- Application
- Production Machine Learning model

For example:

```text
Trained Churn Model
        ↓
Production System
        ↓
New Customer Data
        ↓
Churn Prediction
        ↓
Business Action
```

Deployment is not always a full production Machine Learning service. The appropriate form depends on the project.

---

# 14. Monitoring After Deployment

A deployed solution may need to be monitored.

Reasons include:

- Data can change
- Customer behavior can change
- Business conditions can change
- Model performance can decrease
- Data pipelines can fail

A simplified production cycle is:

```text
Deploy
  ↓
Monitor
  ↓
Evaluate
  ↓
Improve / Retrain
  ↓
Deploy Again
```

This makes Data Science an ongoing process rather than a one-time activity.

---

# 15. Why CRISP-DM is Iterative

One of the most important characteristics of CRISP-DM is that the phases are **iterative**.

For example, after exploring the data, the team may discover:

```text
Important data is missing
```

The team may need to return to:

```text
Data Understanding
        ↓
Data Preparation
```

Or, after evaluation, the team may discover that the model does not solve the business objective.

The team may then revisit:

```text
Evaluation
    ↓
Business Understanding
```

Therefore:

```text
Business Understanding
        ↕
Data Understanding
        ↕
Data Preparation
        ↕
Modeling
        ↕
Evaluation
        ↕
Deployment
```

The arrows represent the possibility of moving backward and forward.

---

# 16. Complete CRISP-DM Workflow

A useful simplified representation is:

```text
       ┌───────────────────────┐
       │ Business Understanding│
       └───────────┬───────────┘
                   ↓
       ┌───────────────────────┐
       │ Data Understanding    │
       └───────────┬───────────┘
                   ↓
       ┌───────────────────────┐
       │ Data Preparation      │
       └───────────┬───────────┘
                   ↓
       ┌───────────────────────┐
       │ Modeling              │
       └───────────┬───────────┘
                   ↓
       ┌───────────────────────┐
       │ Evaluation            │
       └───────────┬───────────┘
                   ↓
       ┌───────────────────────┐
       │ Deployment            │
       └───────────────────────┘
                   ↺
             Iteration / Review
```

---

# 17. Practical Example — Customer Churn

Suppose a telecom company wants to reduce customer churn.

## 17.1 Business Understanding

Goal:

```text
Reduce customer churn.
```

Possible Data Science objective:

```text
Identify customers at high risk of churn.
```

---

## 17.2 Data Understanding

Available data:

```text
Age
Contract
Monthly Charges
Usage
Complaints
Support Calls
Churn Status
```

The team examines distributions, missing values, and relationships.

---

## 17.3 Data Preparation

The team may:

- Handle missing values
- Encode categorical variables
- Create useful features
- Remove irrelevant information
- Split the dataset

---

## 17.4 Modeling

Possible models:

```text
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
```

The models can be compared using suitable evaluation procedures.

---

## 17.5 Evaluation

The team evaluates:

```text
Precision
Recall
F1-score
ROC-AUC
```

It also asks:

```text
Can the business actually use these predictions?
```

---

## 17.6 Deployment

The selected model may be integrated into a customer-management system.

```text
New Customer Data
       ↓
Churn Model
       ↓
Risk Prediction
       ↓
Customer Retention Action
```

The model can then be monitored and updated when appropriate.

---

# 18. Advantages of CRISP-DM

### Structured Process

CRISP-DM provides clear phases for organizing a project.

### Business Focus

The process begins with understanding the actual business problem.

### Iterative

Teams can return to earlier phases as new information is discovered.

### Flexible

It can be adapted to different industries and Data Science projects.

### Better Communication

A common process can help different team members understand project progress.

### Reduces Technical Tunnel Vision

It encourages teams to consider business objectives rather than focusing only on model performance.

---

# 19. Limitations of CRISP-DM

### Not a Strict Step-by-Step Recipe

Projects rarely follow the phases in a perfectly linear sequence.

### Can Require Adaptation

Different project types may require additional steps or specialized methods.

### Does Not Define Every Technical Detail

The framework explains the overall process but does not tell the team which algorithm, database, or programming tool must be used.

### Modern Projects May Need Additional Practices

Production-oriented projects may also require:

- Data engineering
- MLOps
- Model monitoring
- Security
- Governance
- Responsible AI practices

---

# 20. CRISP-DM vs a Simple Data Science Workflow

The two can look similar:

```text
CRISP-DM
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment
```

A generic Data Science workflow may be:

```text
Problem Definition
        ↓
Data Collection
        ↓
Cleaning
        ↓
EDA
        ↓
Modeling
        ↓
Evaluation
        ↓
Communication / Deployment
```

The important common principle is that successful Data Science involves both **technical work and problem understanding**.

---

# 21. Key Takeaways

1. **CRISP-DM** stands for Cross-Industry Standard Process for Data Mining.
2. It provides a structured framework for organizing Data Science and Data Mining projects.
3. It has six major phases:
   - Business Understanding
   - Data Understanding
   - Data Preparation
   - Modeling
   - Evaluation
   - Deployment
4. Business Understanding focuses on defining the actual problem and success criteria.
5. Data Understanding focuses on exploring data and assessing its quality.
6. Data Preparation transforms raw data into a suitable form for analysis and modeling.
7. Modeling involves selecting and training appropriate techniques.
8. Evaluation checks both technical performance and whether the solution meets the original objective.
9. Deployment makes results available for practical use.
10. CRISP-DM is **iterative**, so teams can return to earlier phases.
11. The framework is flexible and can be adapted to different industries and project types.
12. Modern production projects may need additional practices such as monitoring, MLOps, governance, security, and responsible AI.

---

# 22. Personal Understanding

After studying the CRISP-DM model, I understand that Data Science is not simply about choosing an algorithm and training a model.

The process should begin by understanding the real-world problem and defining what success means. Only after that should the available data be examined, cleaned, and prepared.

Modeling comes after the team understands the data, and evaluation should check both model performance and whether the result actually solves the original problem.

I also understand that CRISP-DM is not a rigid sequence. A Data Scientist may need to go backward when new information is discovered. For example, a modeling problem may reveal poor data quality, requiring the team to return to data preparation.

The most important idea is:

> **CRISP-DM provides a structured but iterative framework that connects business objectives, data, modeling, evaluation, and deployment into one Data Science process.**

---

# 23. Interview / Viva Questions

### Q1. What does CRISP-DM stand for?

**Answer:**  
CRISP-DM stands for **Cross-Industry Standard Process for Data Mining**.

### Q2. What are the six phases of CRISP-DM?

**Answer:**  
Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment.

### Q3. What is Business Understanding?

**Answer:**  
It is the phase where the project problem, objectives, success criteria, constraints, and business context are defined.

### Q4. What happens during Data Understanding?

**Answer:**  
The team collects or accesses initial data, explores variables and distributions, identifies data-quality issues, and develops an understanding of the available data.

### Q5. What is Data Preparation?

**Answer:**  
It involves cleaning, transforming, integrating, selecting, and engineering data so that it is suitable for analysis or modeling.

### Q6. What happens during Modeling?

**Answer:**  
Appropriate analytical or Machine Learning techniques are selected, trained, tuned, and compared.

### Q7. What is the purpose of the Evaluation phase?

**Answer:**  
Evaluation checks whether the model performs adequately and whether the overall solution meets the original business or project objective.

### Q8. What is Deployment?

**Answer:**  
Deployment is the process of making the analysis, model, or resulting solution available for practical use.

### Q9. Is CRISP-DM a linear process?

**Answer:**  
No. CRISP-DM is iterative. Teams can return to earlier phases whenever new information or problems are discovered.

### Q10. Why is Business Understanding important?

**Answer:**  
It ensures that the Data Science work is connected to a meaningful real-world objective rather than focusing only on technical model performance.

---

# 24. Conclusion

The CRISP-DM model provides a useful framework for organizing Data Science and Data Mining projects.

Its six phases are:

```text
1. Business Understanding
          ↓
2. Data Understanding
          ↓
3. Data Preparation
          ↓
4. Modeling
          ↓
5. Evaluation
          ↓
6. Deployment
```

However, the process is not strictly linear:

```text
Discover Something New
        ↓
Return to Earlier Phase
        ↓
Improve the Solution
```

CRISP-DM helps ensure that a project connects:

```text
Business Problem
      ↓
Data
      ↓
Analysis / Modeling
      ↓
Evaluation
      ↓
Practical Solution
```

The key lesson is that successful Data Science requires more than technical modeling. Understanding the problem, preparing reliable data, evaluating results correctly, and delivering a useful solution are all essential parts of the process.

---

