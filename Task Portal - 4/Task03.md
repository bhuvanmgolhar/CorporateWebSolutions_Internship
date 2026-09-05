# Task 03 — Security

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal IV |
| Task Number | 03 |
| Topic | Data Science & Artificial Intelligence Security |
| Task Type | Conceptual / Technical & Security |
| Status | Completed |
| Repository Section | `tasks/portal-04/task-03/` |

---

## 2. Objective

The objective of this task is to understand the core principles of **Security in Data Science and Machine Learning Systems**, focusing on data protection, model vulnerabilities, threat vectors, privacy-preserving machine learning (PPML), and enterprise cybersecurity compliance.
This task focuses on:
- Defining data security, model security, and pipeline integrity in Data Science
- Identifying major threat vectors (e.g., Data Poisoning, Model Evasion, Model Inversion, Model Extraction)
- Exploring Privacy-Preserving Machine Learning (PPML) techniques (e.g., Differential Privacy, Federated Learning, Homomorphic Encryption)
- Securing the end-to-end Data Pipeline and MLOps Infrastructure (Access Controls, Encryption, API Security)
- Reviewing security compliance standards and frameworks (e.g., NIST AI RMF, OWASP Top 10 for ML/LLMs, GDPR, ISO 27001)
- Utilizing open-source security toolkits (e.g., Adversarial Robustness Toolbox, PySyft, Opacus)

---

## 3. Introduction

**Data Science Security** addresses the strategies, technical controls, and safeguards required to protect sensitive training data, machine learning pipelines, and trained algorithms from unauthorized access, malicious manipulation, intellectual property theft, and privacy leaks.
As AI and Data Science systems become central to enterprise decision-making, they create novel attack surfaces that traditional software security cannot defend against alone.
A simplified view of security threat propagation in ML pipelines is:

```text
Untrusted Data Sources / Ingestion
        ↓
Data Poisoning & Pipeline Manipulation
        ↓
Vulnerable Model Training & Infrastructure
        ↓
Adversarial Attacks (Evasion / Extraction)
        ↓
Data Leakage, Model Theft & System Compromise

```

Security in Data Science spans two core mandates: **Data Protection** (ensuring confidential records are kept private) and **Model Integrity** (ensuring machine learning models produce accurate, untampered outputs).
The key idea is:

> **Machine learning security must be designed holistically across the entire data lifecycle—from raw data ingestion and model training to deployment and API endpoint monitoring.**

---

# 4. What is Security in Data Science?

## Definition

Security in Data Science encompasses two interconnected domains:

```text
                               Security Domains
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Data Security & Privacy               │ Model & Pipeline Security             │
│ Protecting raw datasets from unauthorized│ Safeguarding algorithms and MLOps     │
│ access, breaches, and privacy leaks.  │ against manipulation, theft & evasion.│
└───────────────────────────────────────┴───────────────────────────────────────┘

```

1. **Data Security & Privacy:** Safeguarding training data at rest, in transit, and during compute using encryption, anonymization, and strict access governance.
2. **Model & Pipeline Security:** Protecting ML algorithms, model artifacts, and deployment infrastructure against adversarial attacks, prompt injection, model inversion, and pipeline compromise.

A simplified overview of secure data science operations is:

```text
Secure Ingestion → Encrypted Storage → Privacy-Preserving Training → Hardened Model API → Active Threat Monitoring

```

---

# 5. Why Data Science Security is Critical

Securing data science systems is mandatory due to unique risks inherent to statistical and machine learning workflows:

```text
Insecure ML System
  ↓
Adversarial Attack / Data Leak
  ↓
Intellectual Property Loss & Regulatory Penalties
  ↓
System Failure & Reputational Damage

```

Key reasons why security is critical in Data Science:

* **Protection of Sensitive Data:** Datasets often contain Personally Identifiable Information (PII), medical records, or proprietary trade secrets.
* **Intellectual Property Protection:** Enterprise machine learning models represent significant research investment and operational value that must be protected from model stealing.
* **System Integrity:** Adversarial manipulation of ML models in high-stakes environments (e.g., healthcare diagnostics, autonomous driving, fraud detection) can cause real-world harm.
* **Regulatory Compliance:** Strict data protection laws (e.g., GDPR, HIPAA, CCPA) penalize unauthorized data processing and security failures.

---

# 6. Major Attack Vectors in Machine Learning

Machine learning models introduce distinct mathematical and algorithmic vulnerabilities that differ from classic software bugs:

```text
ML Attack Vectors
├── Data Poisoning Attacks (Corrupting training data)
├── Model Evasion / Adversarial Attacks (Tricking deployed models)
├── Model Inversion & Inference (Reconstructing training data)
└── Model Stealing / Extraction (Duplicating proprietary models)

```

| Attack Type | Target Stage | Mechanism / Description | Real-World Impact |
| --- | --- | --- | --- |
| **Data Poisoning** | Training Phase | Attacker injects malicious or mislabeled samples into training data to create backdoors or degrade accuracy. | Spam filter trained on poisoned emails misses malicious phishing campaigns. |
| **Model Evasion (Adversarial Input)** | Inference Phase | Attacker adds subtle, human-imperceptible perturbations to input data to force misclassification. | Autonomous vehicle misinterpreting a modified stop sign as a speed limit sign. |
| **Model Inversion** | Inference Phase | Attacker queries the model repeatedly with outputs to reconstruct sensitive training data. | Reconstructing patient facial images or genetic profiles from a medical diagnostic AI model. |
| **Membership Inference** | Inference Phase | Attacker determines whether a specific individual's data record was used in the model's training set. | Confirming an individual attended a specific medical clinic based on model output probabilities. |
| **Model Extraction / Stealing** | Inference Phase | Attacker queries a public model API systematically to train a clone model, stealing intellectual property. | Duplicating a proprietary financial prediction API without paying licensing fees. |

---

# 7. Data Privacy and Anonymization Techniques

Before raw data enters a machine learning pipeline, privacy engineering controls must reduce exposure of sensitive identity information.

```text
Data Protection Stack
├── Data Anonymization (Removing direct identifiers)
├── Pseudonymization (Replacing identifiers with token keys)
├── Data Masking & Redaction (Obscuring sensitive fields)
└── Cryptographic Encryption (AES-256 at rest, TLS 1.3 in transit)

```

## Traditional Privacy Techniques

* **K-Anonymity:** Ensures that any individual in a dataset cannot be distinguished from at least $k-1$ other individuals whose records also appear in the dataset.
* **L-Diversity:** Extends k-anonymity by ensuring sensitive attributes within each anonymized group contain at least $l$ distinct well-represented values.
* **T-Closeness:** Ensures the distribution of a sensitive attribute in any group is close to the attribute distribution across the entire dataset.

---

# 8. Privacy-Preserving Machine Learning (PPML)

Modern security engineering leverages cryptographic and statistical frameworks to train machine learning models without exposing raw data.

```text
Advanced PPML Architectures
├── Differential Privacy (DP) (Adding mathematical noise)
├── Federated Learning (FL) (Decentralized model training)
├── Homomorphic Encryption (HE) (Computing over encrypted data)
└── Secure Multi-Party Computation (SMPC) (Distributed cryptographic computation)

```

### 1. Differential Privacy (DP)

Differential privacy mathematically guarantees that adding or removing a single individual's record from a dataset does not significantly change the outcome of model training. It works by injecting controlled statistical noise (e.g., Gaussian or Laplacian noise) into queries or gradients during training.

### 2. Federated Learning (FL)

A decentralized learning approach where local edge devices (e.g., smartphones or local hospital servers) train models locally on their own data and share only model gradient updates with a central server, keeping raw data local.

### 3. Homomorphic Encryption (HE)

A form of encryption that allows mathematical computations (e.g., neural network matrix multiplications) to be performed directly on encrypted ciphertexts without decrypting them first.

---

# 9. Securing the Data Science Infrastructure & MLOps

Security must be embedded into MLOps pipelines (DevSecOps / MLSecOps) to protect code, notebooks, and model artifacts.

```text
MLSecOps Safeguards
├── Role-Based Access Control (RBAC) (Least-privilege access to data lakes)
├── Secure Code Environments (Hardening Jupyter Notebooks & IDEs)
├── Model Artifact Signing (Cryptographic hashes to prevent artifact tampering)
└── API Gateway Defense (Rate limiting, authentication, and input validation)

```

## Essential Infrastructure Security Controls

* **Access Control:** Enforcing Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC) across data lakes, S3 buckets, and model registries.
* **Jupyter & Notebook Security:** Disabling root access, enforcing multi-factor authentication (MFA), and eliminating plain-text API secrets from code cells.
* **API Security:** Protecting inference endpoints with rate-limiting, authentication tokens (OAuth2/JWT), and strict input validation layers to block malicious payloads.

---

# 10. Security Frameworks and Compliance Standards

Data science teams must align their security architecture with international cybersecurity frameworks and AI regulations:

| Framework / Standard | Focus Area | Primary Requirements |
| --- | --- | --- |
| **NIST AI Risk Management Framework (AI RMF)** | AI System Security & Safety | Framework to manage risks, enhance trustworthiness, safety, and security in AI. |
| **OWASP Top 10 for ML / LLM** | Vulnerability Awareness | Guidance on top threats like Prompt Injection, Data Poisoning, and Model Denial of Service. |
| **GDPR (General Data Protection Regulation)** | Privacy & Data Rights | Mandates data minimization, right to erasure, and limits unauthorized automated processing. |
| **ISO/IEC 27001 & 42001** | Information Security & AI Governance | Standardized management systems for information security (27001) and AI systems (42001). |

---

# 11. Security Toolkits for Data Science

Data scientists and security engineers utilize specialized open-source tools to audit and harden AI pipelines:

```text
Security Toolkits Ecosystem
├── Adversarial Robustness Toolbox (ART) → Testing defense against adversarial attacks
├── PySyft (OpenMined)                   → Privacy-preserving federated learning
└── Opacus (PyTorch)                     → Differential privacy for deep learning

```

* **Adversarial Robustness Toolbox (ART):** An open-source Python library developed by Linux Foundation AI & Data providing tools to evaluate, defend, and verify machine learning models against evasion, poisoning, extraction, and inversion attacks.
* **Opacus:** A PyTorch library that enables training PyTorch models with Differential Privacy (DP-SGD) with minimal code changes.

---

---

# 25. Personal Understanding

After studying Security in Data Science and Machine Learning, I understand that securing AI systems requires protecting both data privacy and algorithmic integrity.
I recognize that machine learning models introduce unique vulnerabilities—such as data poisoning during training, adversarial evasion during inference, and privacy leaks through model inversion—that conventional network firewalls cannot prevent.
I understand that privacy-preserving techniques like Differential Privacy, Federated Learning, and Homomorphic Encryption allow data scientists to extract valuable insights without exposing raw personal data. Furthermore, integrating security into MLOps (MLSecOps) ensures access control, pipeline auditing, and API protection throughout the AI lifecycle.
The key takeaway is:

> **Security is not an afterthought in Data Science; it is a fundamental architectural requirement designed to ensure data privacy, model robustness, and regulatory compliance.**

---

# 26. Interview / Viva Questions

### Q1. What is Data Science Security?

**Answer:**

Data Science Security refers to the practices, technical controls, and cryptographic techniques used to protect raw datasets, training pipelines, machine learning models, and deployment infrastructure from unauthorized access, tampering, and privacy leaks.

### Q2. What is Data Poisoning?

**Answer:**

Data Poisoning is an attack occurring during the training phase where an attacker maliciously alters or injects corrupt samples into the training dataset to compromise model accuracy or insert a backdoor.

### Q3. What is an Adversarial Evasion Attack?

**Answer:**

An Adversarial Evasion Attack involves modifying input data with subtle, often imperceptible perturbations during inference to trick a trained machine learning model into making wrong predictions.

### Q4. What is Model Inversion?

**Answer:**

Model Inversion is a privacy attack where an adversary queries a trained model repeatedly with outputs to reconstruct sensitive features or images from the underlying training dataset.

### Q5. What is Membership Inference?

**Answer:**

Membership Inference is an attack that determines whether a specific data record was part of a machine learning model's training dataset by analyzing the model's prediction output confidence scores.

### Q6. What is Differential Privacy?

**Answer:**

Differential Privacy is a mathematical framework that guarantees privacy by injecting controlled noise into query results or training gradients, ensuring an individual's presence in a dataset cannot be reliably inferred.

### Q7. How does Federated Learning enhance data privacy?

**Answer:**

Federated Learning keeps raw training data stored locally on edge devices, sending only encrypted gradient updates to a central server, reducing centralized data exposure risks.

### Q8. What is Homomorphic Encryption?

**Answer:**

Homomorphic Encryption is a form of cryptography that enables mathematical computations to be executed directly on encrypted data without needing to decrypt it first.

### Q9. What is K-Anonymity?

**Answer:**

K-Anonymity is a data privacy property where any individual's record in a released dataset is mathematically indistinguishable from at least $k-1$ other records with respect to quasi-identifiers.

### Q10. What is Model Extraction (Model Stealing)?

**Answer:**

Model Extraction is an attack where an adversary queries a public model API systematically with varied inputs to train a surrogate model that replicates the proprietary target model's functionality.

### Q11. What is MLSecOps?

**Answer:**

MLSecOps (Machine Learning Security Operations) integrates security practices, access governance, and vulnerability testing into the MLOps lifecycle from data ingestion to continuous model monitoring.

### Q12. What is the role of the Adversarial Robustness Toolbox (ART)?

**Answer:**

ART is an open-source Python library used to test, evaluate, and defend machine learning models against adversarial evasion, poisoning, extraction, and inversion threats.

### Q13. Why are Jupyter Notebooks considered security risks if unhardened?

**Answer:**

Unsecured Jupyter Notebooks may execute arbitrary code, expose hardcoded API keys or database credentials, and lack role-based access controls if exposed directly to external networks.

### Q14. What is Prompt Injection in Large Language Models (LLMs)?

**Answer:**

Prompt Injection is a vulnerability where malicious user input overrides system instructions in an LLM, causing the model to perform unintended actions or expose confidential context data.

### Q15. How does encryption at rest protect data science workloads?

**Answer:**

Encryption at rest uses strong algorithms (like AES-256) to protect stored training datasets, database backups, and model weight artifacts from unauthorized access in the event of physical or cloud storage breaches.

---

# 27. Conclusion

Data Science Security bridges cybersecurity engineering, cryptography, and machine learning theory.
Its basic workflow can be summarized as:

```text
Secure Raw Data Ingestion
      ↓
Privacy Protection (Anonymization / Differential Privacy)
      ↓
Encrypted Infrastructure & Secure Pipeline Training
      ↓
Model Hardening & Adversarial Auditing
      ↓
Secured API Endpoints & Continuous Monitoring
      ↓
Trustworthy & Compliant AI Operations

```

The major components include:

```text
Data Science Security
├── Data Privacy & Anonymization (Differential Privacy, K-Anonymity)
├── Model Vulnerability Mitigation (Poisoning, Evasion, Inversion)
├── Privacy-Preserving ML (Federated Learning, Homomorphic Encryption)
└── MLSecOps & Compliance (NIST AI RMF, OWASP Top 10, GDPR)

```

Core tools and technologies include:

```text
Adversarial Robustness Toolbox (ART) / Opacus / PySyft
AES-256 / TLS 1.3 / Homomorphic Encryption
Differential Privacy (DP-SGD)
Role-Based Access Control (RBAC) & OAuth2
NIST AI RMF & OWASP Standards

```

Prioritizing security throughout the data science lifecycle ensures that enterprise machine learning applications remain resilient against emerging threats, protect user privacy, and satisfy strict regulatory requirements.
The key takeaway is:

> **Securing data science systems requires proactive defenses that protect sensitive training data, secure MLOps infrastructure, and harden machine learning models against adversarial manipulation.**

---

---

# 30. Key Takeaways

1. **Data Science Security safeguards raw datasets, training pipelines, and ML models from malicious threats.**
2. Security spans two core pillars: Data Protection (privacy) and Model Integrity (robustness).
3. **Data Poisoning** attacks corrupt training datasets to degrade accuracy or insert malicious backdoors.
4. **Adversarial Evasion** attacks apply subtle perturbations to inputs during inference to fool model predictions.
5. **Model Inversion** queries models to mathematically reconstruct sensitive training samples.
6. **Membership Inference** determines whether an individual's data record was used during model training.
7. **Model Stealing** extracts proprietary model functionality through systematic public API querying.
8. **Differential Privacy** adds statistical noise to data or gradients, providing mathematical privacy guarantees.
9. **Federated Learning** trains models locally across distributed edge devices without centralizing raw data.
10. **Homomorphic Encryption** enables model computation directly on encrypted ciphertexts.
11. **K-Anonymity** ensures an individual's record cannot be distinguished from at least $k-1$ other records.
12. **MLSecOps** embeds security automation, access control, and vulnerability testing into MLOps pipelines.
13. Unhardened Jupyter Notebooks and plaintext API keys represent major operational security vulnerabilities.
14. The **NIST AI RMF** and **OWASP Top 10 for LLM/ML** provide guidelines for securing AI applications.
15. **Adversarial Robustness Toolbox (ART)** is an open-source suite for testing model defenses against attacks.
16. Regulatory frameworks like **GDPR** enforce data minimization, erasure rights, and strict privacy controls.
17. Building secure AI requires continuous threat modeling, vulnerability auditing, and access control.

---

# 31. Personal Understanding

After studying Security in Data Science and Machine Learning, I understand that securing AI systems requires protecting both data privacy and algorithmic integrity.
I recognize that machine learning models introduce unique vulnerabilities—such as data poisoning during training, adversarial evasion during inference, and privacy leaks through model inversion—that conventional network firewalls cannot prevent.
I understand that privacy-preserving techniques like Differential Privacy, Federated Learning, and Homomorphic Encryption allow data scientists to extract valuable insights without exposing raw personal data. Furthermore, integrating security into MLOps (MLSecOps) ensures access control, pipeline auditing, and API protection throughout the AI lifecycle.
The ultimate lesson is:

> **Securing data science systems requires proactive defenses that protect sensitive training data, secure MLOps infrastructure, and harden machine learning models against adversarial manipulation.**

---

# 32. Interview / Viva Questions

### Q1. What is the primary difference between traditional software security and machine learning security?

**Answer:**

Traditional software security focuses on code vulnerabilities, network perimeters, and access controls. Machine learning security must also defend against data-driven attacks like adversarial input manipulation, training data poisoning, and model inversion.

### Q2. How can an attacker execute a Data Poisoning attack?

**Answer:**

An attacker injects corrupted, mislabeled, or crafted malicious samples into an unvetted public training dataset, causing the trained model to fail on specific inputs or exhibit backdoors.

### Q3. What is an Adversarial Example in machine learning?

**Answer:**

An Adversarial Example is an input modified with small, calculated noise that causes a machine learning model to misclassify it with high confidence while remaining unchanged to human perception.

### Q4. What is the difference between Model Inversion and Membership Inference?

**Answer:**

Model Inversion attempts to reconstruct the actual training data or features. Membership Inference determines whether a specific known data record was included in the model's training set.

### Q5. What is Privacy-Preserving Machine Learning (PPML)?

**Answer:**

PPML is a set of techniques (such as Differential Privacy, Federated Learning, and Homomorphic Encryption) designed to train and execute ML models without exposing raw sensitive underlying data.

### Q6. How does Differential Privacy protect against privacy leaks?

**Answer:**

By injecting controlled random noise into query computations or training gradients (DP-SGD), preventing an adversary from determining whether any single individual's record was used.

### Q7. What is Federated Learning?

**Answer:**

A decentralized framework where edge devices train local models on local data and transmit only encrypted model updates to a central server for aggregation.

### Q8. What is Secure Multi-Party Computation (SMPC)?

**Answer:**

A subfield of cryptography that enables multiple parties to jointly compute a function over their inputs while keeping those inputs private from each other.

### Q9. What is Model Stealing?

**Answer:**

An attack where an adversary queries a target model API repeatedly to collect input-output pairs and train a surrogate model that mirrors the original model's performance.

### Q10. What is the OWASP Top 10 for Large Language Models?

**Answer:**

A community-driven standard highlighting the top security vulnerabilities in LLMs, including Prompt Injection, Data Poisoning, Model Denial of Service, and Supply Chain Vulnerabilities.

### Q11. What is Role-Based Access Control (RBAC) in Data Science?

**Answer:**

A security practice that restricts data lake and model registry access based on an employee's organizational role, enforcing least-privilege principles.

### Q12. What is Data Anonymization?

**Answer:**

The process of removing or modifying Personally Identifiable Information (PII) from datasets so that individual data subjects cannot be identified.

### Q13. Why are hardcoded credentials in ML notebooks hazardous?

**Answer:**

Hardcoded credentials (e.g., API keys, database passwords) can be inadvertently leaked when notebooks are committed to public version control repositories like GitHub.

### Q14. What is DP-SGD?

**Answer:**

Differentially Private Stochastic Gradient Descent (DP-SGD) is a model training algorithm that clips gradients and adds noise during optimization to ensure differential privacy guarantees.

### Q15. How does rate-limiting protect ML inference APIs?

**Answer:**

Rate-limiting restricts the number of API requests a user can make in a given timeframe, mitigating automated model extraction, inversion, and denial-of-service attacks.

### Q16. What is the NIST AI Risk Management Framework?

**Answer:**

A guidance document created by NIST to help organizations manage risks, ensure safety, enhance trustworthiness, and enforce security across AI system lifecycles.

### Q17. What is the ultimate goal of Data Science Security?

**Answer:**

To ensure data confidentiality, protect intellectual property, uphold data privacy rights, and guarantee that ML models operate reliably, securely, and free from adversarial compromise.

---

# 33. Conclusion

Data Science Security is a essential discipline ensuring that AI systems remain trustworthy, private, and resilient against malicious compromise.
Its basic workflow can be represented as:

```text
Data Protection & Privacy Engineering
      ↓
Secure Ingestion & Encrypted Storage
      ↓
Privacy-Preserving Training (DP / Federated Learning)
      ↓
Model Hardening & Vulnerability Auditing
      ↓
MLSecOps Infrastructure & API Defense
      ↓
Resilient & Secure AI Systems

```

The major model categories are:

```text
Data Science Security
├── Data Protection & Privacy (Differential Privacy, Anonymization)
├── Model Defense & Vulnerabilities (Poisoning, Evasion, Inversion)
├── Privacy-Preserving Architectures (Federated Learning, Homomorphic Encryption)
└── MLSecOps & Compliance (NIST AI RMF, OWASP Top 10, GDPR)

```

Important technologies and concepts include:

```text
Adversarial Robustness Toolbox (ART) / Opacus / PySyft
AES-256 / TLS 1.3 / Homomorphic Encryption
Differential Privacy (DP-SGD)
Role-Based Access Control (RBAC)
NIST AI RMF & OWASP Compliance Standards

```

Adopting proactive security measures throughout the data science lifecycle protects enterprise assets, ensures regulatory compliance, and fosters trust in automated decision-making systems.
The most important lesson is:

> **Securing data science systems requires proactive defenses that protect sensitive training data, secure MLOps infrastructure, and harden machine learning models against adversarial manipulation.**
