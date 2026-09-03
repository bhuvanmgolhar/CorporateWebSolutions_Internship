# Task 06 — The Role of Questions in Data Science

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal II |
| Task Number | 06 |
| Topic | The Role of Questions in Data Science |
| Task Type | Conceptual / Problem-Solving |
| Status | Completed |
| Repository Section | `tasks/portal-02/task-06/` |

---

## 2. Objective

The objective of this task is to understand why **asking the right questions is one of the most important parts of Data Science**.

This task focuses on:

- Understanding the role of questions in Data Science
- Converting a vague problem into clear analytical questions
- Distinguishing business questions from data questions
- Understanding descriptive, diagnostic, predictive, and prescriptive questions
- Learning how good questions guide data collection and analysis
- Understanding how poor questions can lead to irrelevant results
- Connecting questions with the Data Science workflow

---

## 3. Introduction

Data Science does not begin with a dataset or a Machine Learning algorithm.

It begins with a **question or problem**.

A Data Scientist should first understand:

```text
What are we trying to find out?
              ↓
Why do we need to know it?
              ↓
What data can help answer it?
              ↓
What analysis is appropriate?
              ↓
What action can result?
```

A useful Data Science project therefore starts with a clearly defined question.

The key idea is:

> **A good Data Science solution depends on asking the right question before analyzing the data.**

---

# 4. Why Questions Matter in Data Science

Organizations often have broad statements such as:

```text
"Sales are falling."
"Customers are leaving."
"We have too much inventory."
"Our website performance is poor."
```

These are problems, but they are not yet precise analytical questions.

A Data Scientist needs to transform them into questions that can be investigated using data.

For example:

```text
Broad Problem:
Sales are falling.

Better Questions:
Which products experienced the largest decline?
Which regions experienced the decline?
When did the decline begin?
Are changes in price or promotions related to the decline?
```

The quality of these questions influences the direction of the entire analysis.

---

# 5. Characteristics of a Good Data Science Question

A good question should generally be:

### Clear

Everyone involved should understand what is being asked.

### Specific

The question should define the subject, time period, population, or other relevant boundaries where appropriate.

### Relevant

It should address a meaningful problem.

### Answerable

There should be data or a realistic way to obtain evidence needed to investigate it.

### Measurable

The result should be expressible or evaluated using appropriate measurements.

### Actionable

Where possible, the answer should support a decision or useful next step.

A useful pattern is:

```text
Clear + Relevant + Answerable + Measurable
                    ↓
            Useful Question
```

---

# 6. Business Questions vs Data Questions

These are related but different.

## Business Question

A business question focuses on the organization's objective.

Example:

```text
How can we reduce customer churn?
```

## Data Question

A data question focuses on what can be investigated using data.

Examples:

```text
Which customer groups have the highest churn rate?

Which behaviors are associated with churn?

Can we predict which customers are at higher risk?
```

The transition is:

```text
Business Problem
      ↓
Business Question
      ↓
Data Question
      ↓
Analysis / Model
```

---

# 7. Types of Questions in Data Science

A useful framework is to classify questions into four broad types:

```text
Descriptive
   ↓
Diagnostic
   ↓
Predictive
   ↓
Prescriptive
```

These categories are related but answer different kinds of questions.

---

# 8. Descriptive Questions

## Definition

**Descriptive questions ask what happened or what the data currently shows.**

Examples:

```text
What were total sales last month?
Which product sold the most?
How many customers churned?
What is the average order value?
```

These questions focus on summarizing existing data.

## Common Techniques

- Aggregation
- Grouping
- Summary statistics
- Dashboards
- Visualization

Example:

```text
Sales Data
   ↓
Group by Region
   ↓
Calculate Revenue
   ↓
Regional Sales Report
```

---

# 9. Diagnostic Questions

## Definition

**Diagnostic questions ask why something happened or what factors may be associated with an observed outcome.**

Examples:

```text
Why did sales decrease?
Why did customer churn increase?
Which factors are associated with lower conversion?
```

Possible techniques include:

- Comparisons
- Correlation analysis
- Segmentation
- Statistical analysis
- Drill-down analysis
- Experiments where appropriate

Important point:

> **Association does not automatically prove causation.**

For example, if two variables move together, more investigation may be required before claiming that one caused the other.

---

# 10. Predictive Questions

## Definition

**Predictive questions ask what is likely to happen in the future or what outcome may occur for a new observation.**

Examples:

```text
Which customers are likely to churn?
What will next month's sales be?
What is the expected demand for a product?
Is this transaction likely to be fraudulent?
```

Machine Learning can be useful for many predictive questions.

A simplified workflow is:

```text
Historical Data
      ↓
Known Outcomes
      ↓
Train Model
      ↓
New Data
      ↓
Prediction
```

---

# 11. Prescriptive Questions

## Definition

**Prescriptive questions ask what action should be taken based on available evidence, predictions, constraints, and objectives.**

Examples:

```text
Which products should we promote?
How much inventory should we order?
Which customers should receive an offer?
What action could reduce expected losses?
```

Prescriptive analysis may combine:

- Predictions
- Optimization
- Business rules
- Constraints
- Domain knowledge

A prediction does not automatically determine the correct action.

---

# 12. Example: Customer Churn

Consider a telecom company.

## Descriptive

```text
What percentage of customers churned last quarter?
```

## Diagnostic

```text
Which customer characteristics are associated with higher churn?
```

## Predictive

```text
Which current customers are most likely to churn?
```

## Prescriptive

```text
What retention action should be considered for high-risk customers?
```

This progression shows how one business problem can lead to different types of Data Science questions.

---

# 13. Turning a Vague Question into a Good Question

Suppose the business says:

```text
"Customers are unhappy."
```

This is too broad.

A Data Scientist might ask:

```text
Which customers are reporting dissatisfaction?
```

Then:

```text
What issues are they reporting?
```

Then:

```text
When did dissatisfaction increase?
```

Then:

```text
Which customer segments have the highest complaint rate?
```

Then potentially:

```text
Can we predict which customers are at high risk of dissatisfaction?
```

The questions become progressively more specific and measurable.

---

# 14. Questions Guide Data Collection

The question helps determine what data is needed.

For example:

```text
Question:
Why are customers cancelling subscriptions?
```

Potentially relevant data may include:

- Cancellation date
- Subscription duration
- Customer activity
- Support interactions
- Pricing
- Product usage
- Customer feedback

Without a clear question, organizations may collect large amounts of data without knowing how it will be used.

Therefore:

> **The question helps determine what data is relevant.**

---

# 15. Questions Guide Analysis

Different questions require different analytical methods.

```text
Question
   ↓
Type of Problem
   ↓
Appropriate Analysis
```

Examples:

```text
"What happened?"
→ Descriptive Analysis

"Why might it have happened?"
→ Diagnostic Analysis

"What is likely to happen?"
→ Predictive Modeling

"What should we do?"
→ Prescriptive Analysis / Decision Analysis
```

This prevents the common mistake of choosing a method first and looking for a question afterward.

---

# 16. Questions and Machine Learning

Machine Learning should not be the starting point just because an ML algorithm is available.

A better sequence is:

```text
Problem
   ↓
Question
   ↓
Data
   ↓
Analysis
   ↓
ML if Appropriate
   ↓
Evaluation
   ↓
Action / Decision
```

For example, if the question is simply:

```text
What were last month's sales?
```

a basic database query or aggregation may be enough.

There is no reason to build a Machine Learning model for every question.

---

# 17. Primary Questions and Follow-Up Questions

A project often begins with one primary question and then produces additional questions.

Example:

### Primary Question

```text
Why are sales declining?
```

This may lead to:

```text
Which products are declining?
Which stores are affected?
When did the decline start?
Are promotions changing?
Are prices changing?
Is customer traffic changing?
```

These follow-up questions help narrow the investigation.

---

# 18. Asking Questions About Data Quality

Data Scientists should also question the data itself.

Examples:

```text
Where did this data come from?
How was it collected?
Is the data complete?
Are there missing values?
Are labels reliable?
Are there duplicate records?
Does the dataset represent the population we care about?
```

These questions are important because poor-quality or biased data can lead to misleading results.

---

# 19. Questions About Assumptions

Data Science projects contain assumptions.

A team should ask:

```text
Are we measuring the right outcome?
Are the variables defined correctly?
Are historical patterns still relevant?
Are we comparing similar groups?
Could another factor explain the observed relationship?
```

Questioning assumptions helps reduce the risk of drawing incorrect conclusions.

---

# 20. Question Quality and Decision Quality

A useful chain is:

```text
Question
   ↓
Data
   ↓
Analysis
   ↓
Finding
   ↓
Decision
   ↓
Action
```

An issue at the beginning can affect everything that follows.

For example:

```text
Wrong Question
      ↓
Irrelevant Data
      ↓
Incorrect Analysis
      ↓
Misleading Finding
      ↓
Poor Decision
```

Therefore, careful question formulation is an important part of Data Science quality.

---

# 21. Practical Example — E-Commerce Sales

Suppose an online store says:

```text
"We want to improve sales."
```

Possible questions are:

### Descriptive

```text
Which products generate the most revenue?
```

### Diagnostic

```text
Why are certain product categories underperforming?
```

### Predictive

```text
What products are likely to have high demand next month?
```

### Prescriptive

```text
Which products should receive additional promotional attention?
```

Each question requires a different type of analysis.

---

# 22. Questions in the CRISP-DM Process

Questions are connected to multiple phases of CRISP-DM.

```text
Business Understanding
→ What problem are we solving?

Data Understanding
→ What data do we have?

Data Preparation
→ Is the data suitable?

Modeling
→ Can a model answer the question?

Evaluation
→ Did we answer the question reliably?

Deployment
→ How will the answer be used?
```

This shows that questioning is not limited to the beginning of a project.

---

# 23. Stakeholder Questions

Data Scientists should communicate with stakeholders rather than making assumptions about what they want.

Useful questions include:

- What decision will this analysis support?
- Who will use the results?
- What does success look like?
- What constraints exist?
- Which errors are most costly?
- What is the time period of interest?
- What action will be possible after the analysis?

These questions help connect technical work with the real objective.

---

# 24. Common Mistakes When Asking Questions

### Too Broad

```text
"Why are sales bad?"
```

Better:

```text
"Which products and regions contributed most to the decline in sales during the last quarter?"
```

### Too Technical

Starting with:

```text
"Which neural network should we use?"
```

before defining the problem.

### Not Actionable

A question may produce an interesting result but have no practical use.

### Ignoring Context

A question should consider the business, scientific, or operational context.

### Assuming Causation

A correlation or prediction should not automatically be interpreted as proof of causation.

---

# 25. A Simple Question Framework

Before beginning an analysis, ask:

```text
1. What do we want to know?
2. Why do we need to know it?
3. Who will use the answer?
4. What data can answer it?
5. Is the data reliable?
6. What type of analysis is appropriate?
7. How will success be measured?
8. What action could follow?
```

This creates a strong connection between the question, analysis, and final outcome.

---

# 26. Key Takeaways

1. **Good Data Science starts with good questions.**
2. A broad business problem should be converted into clear, specific, and measurable questions.
3. Business questions and data questions are related but serve different purposes.
4. Descriptive questions ask what happened.
5. Diagnostic questions investigate why something happened or what factors are associated with it.
6. Predictive questions ask what is likely to happen.
7. Prescriptive questions ask what action should be considered.
8. Questions help determine which data is needed.
9. Questions also help determine which analytical or Machine Learning methods are appropriate.
10. Machine Learning should not be used simply because it is available.
11. Data Scientists should question the quality of data and the assumptions behind an analysis.
12. Good questions can improve the relevance and usefulness of the final decision.
13. Asking questions is an ongoing activity throughout the Data Science lifecycle, not just the first step.

---

# 27. Personal Understanding

After studying the role of questions in Data Science, I understand that the quality of an analysis depends heavily on how clearly the problem is defined.

A Data Scientist should not immediately start cleaning data or building a Machine Learning model without understanding what the organization actually wants to know.

Questions help convert a vague problem into something that can be investigated using data. They also help determine which data is relevant, which analysis should be performed, and how the final result will be used.

I also understand the difference between descriptive, diagnostic, predictive, and prescriptive questions. These categories help identify the purpose of an analysis and prevent using unnecessarily complex methods.

The most important idea is:

> **The right question guides the right data, the right analysis, and ultimately the right decision.**

---

# 28. Interview / Viva Questions

### Q1. Why are questions important in Data Science?

**Answer:**  
Questions define what the analysis is trying to discover and help determine the relevant data, methods, evaluation criteria, and potential decisions.

### Q2. What is a descriptive question?

**Answer:**  
A descriptive question asks what happened or what the data currently shows.

### Q3. What is a diagnostic question?

**Answer:**  
A diagnostic question investigates why something happened or which factors may be associated with an observed outcome.

### Q4. What is a predictive question?

**Answer:**  
A predictive question asks what is likely to happen in the future or what outcome is likely for a new observation.

### Q5. What is a prescriptive question?

**Answer:**  
A prescriptive question asks what action should be considered based on predictions, evidence, constraints, and objectives.

### Q6. Give an example of a business question.

**Answer:**  
"How can we reduce customer churn?" is an example of a business question.

### Q7. Give an example of a data question.

**Answer:**  
"Which customer characteristics are associated with higher churn?" is an example of a data question.

### Q8. Why should we not start with a Machine Learning algorithm?

**Answer:**  
The algorithm should be selected based on the problem and question. Starting with an algorithm can result in solving an irrelevant problem or using unnecessary complexity.

### Q9. How do questions help with data collection?

**Answer:**  
A clear question helps identify which variables, records, time periods, and data sources are relevant to the analysis.

### Q10. Can correlation prove causation?

**Answer:**  
No. Correlation indicates an association between variables, but additional evidence and appropriate study designs are generally needed to establish causation.

---

# 29. Conclusion

Questions are one of the foundations of effective Data Science.

A useful process is:

```text
Real-World Problem
        ↓
Clear Question
        ↓
Relevant Data
        ↓
Appropriate Analysis
        ↓
Reliable Finding
        ↓
Decision / Action
```

The four broad question types provide a useful framework:

```text
Descriptive  → What happened?
Diagnostic   → Why did it happen?
Predictive   → What is likely to happen?
Prescriptive → What should we consider doing?
```

Good questions also help teams evaluate data quality, challenge assumptions, select suitable methods, and communicate objectives.

The most important lesson is:

> **Data Science is not simply about finding patterns in data. It is about using data to answer meaningful questions and support better decisions.**

---

