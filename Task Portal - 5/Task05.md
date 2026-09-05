# Task 05 — Open-Source & Third-Party Public Datasets: Acquisition, Licensing, Lineage & Quality Audit

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal V |
| Task Number | 05 |
| Topic | External Data — Acquisition, API Integration, Licensing Compliance, Provenance & Quality Auditing |
| Task Type | Technical Core & Enterprise Data Engineering |
| Status | Completed |
| Repository Section | `tasks/portal-05/task-05/` |

---

## 2. Objective

The objective of this task is to construct a production-ready framework for acquiring, validating, integrating, and legally governing **Open-Source & Third-Party External Datasets** for enterprise analytics and machine learning workflows.
This task focuses on:
- Designing automated data ingestion architecture across REST/GraphQL APIs, web scraping, open data portals, and cloud object stores.
- Evaluating open-source data licenses (MIT, Apache 2.0, CC-BY, CC0, GPL) to prevent legal compliance violations in commercial software.
- Tracking data provenance, version control, and immutability using cryptographic hashing and open metadata standards.
- Detecting distribution drift (Covariate Shift, Prior Probability Shift, Concept Shift) between external public datasets and internal production distributions.
- Establishing automated quality auditing pipelines using statistical testing (Kolmogorov-Smirnov, Population Stability Index, Chi-Square) to identify invalid external records.

---

## 3. Introduction

External public datasets—such as government statistics, academic benchmark repositories, satellite imagery, public economic indicators, and web-scraped content—allow enterprises to enrich internal models without high annotation costs.

```text
                     External Data Ingestion Ecosystem
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ External Public  │ ───► │ Automated        │ ───► │ License & Compliance│
│ Data Sources     │      │ Ingestion Hub    │      │ Legal Audit      │
└──────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                             │
┌──────────────────┐      ┌──────────────────┐               │
│ Enriched ML      │ ◄─── │ Statistical Drift│ ◄─────────────┘
│ Training Pipeline│      │ & Quality Audit  │
└──────────────────┘      └──────────────────┘

```

However, uncurated external datasets introduce legal risks (e.g., license mismatches, copyright infringement) and data drift issues that can degrade model performance.
The core principle governing external dataset utilization is:

> **External data accelerates model capabilities only when ingested through automated, cryptographically traceable pipelines that enforce strict licensing compliance and distribution verification.**

---

## 4. Open-Source Data Licensing & Legal Compliance Framework

Integrating third-party datasets into commercial products requires strict adherence to open-source licenses to avoid intellectual property risks.

```text
                       Open Data Licensing Spectrum
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ License Type                          │ Enterprise Commercial Suitability     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ CC0 / Public Domain / PDDL            │ Fully Permissive: Free commercial use,│
│                                       │ modification, and redistribution.     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ MIT / Apache 2.0 / CC-BY 4.0          │ Permissive with Attribution: Requires │
│                                       │ retaining copyright notices.          │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ CC-BY-NC (Non-Commercial)             │ Prohibited for Commercial AI: Strictly│
│                                       │ forbids commercial applications.      │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ ODbL (Open Database License) / GPL    │ Copyleft Hazard: Modifications or     │
│                                       │ derivative datasets must be shared.   │
└───────────────────────────────────────┴───────────────────────────────────────┘

```

### License Audit Verification Protocols

* **Attribution Auditing:** Automated pipelines must generate and store attribution notices for CC-BY and Apache 2.0 compliant assets.
* **Copyleft Isolation:** External datasets licensed under ODbL or GPL must be isolated in separate feature processing layers to prevent triggering derivative-work distribution requirements on proprietary internal models.

---

## 5. Statistical Framework for Dataset Drift & Quality Auditing

External datasets often exhibit distribution shifts compared to internal operational data due to differing collection methodologies, target demographics, or sampling frequencies.

### 5.1 Kolmogorov-Smirnov (K-S) Test for Continuous Feature Shift

The two-sample K-S test evaluates whether an external continuous feature $F_1(x)$ and an internal feature $F_2(x)$ originate from the same continuous distribution:

$$D = \sup_x \vert{}F_1(x) - F_2(x)\vert{}$$

Where:

* $\sup_x$ represents the supremum of the set of distances.
* The null hypothesis $H_0$ (identical distributions) is rejected if $D > c(\alpha) \sqrt{\frac{n_1 + n_2}{n_1 n_2}}$.

---

### 5.2 Population Stability Index (PSI)

Quantifies distribution shift between baseline internal features and external external features across $B$ binned intervals:

$$\text{PSI} = \sum_{b=1}^B \left( \% \text{ Actual}_b - \% \text{ Expected}_b \right) \times \ln\left( \frac{\% \text{ Actual}_b}{\% \text{ Expected}_b} \right)$$

```text
                         PSI Interpretation Matrix
┌─────────────────────────────────────────────────────────────────────────────┐
│ $\text{PSI} < 0.10$       ──► No significant distribution change.           │
│ $0.10 \le \text{PSI} < 0.25$ ──► Slight shift; warrants monitoring.         │
│ $\text{PSI} \ge 0.25$       ──► Significant distribution shift; action required. │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

### 5.3 Categorical Drift: Chi-Square Goodness-of-Fit

Measures deviation between expected internal class proportions $E_i$ and observed external frequencies $O_i$:

$$\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}$$

---

## 6. Data Provenance, Lineage & Cryptographic Immutability

Tracking the source and processing history of external datasets ensures pipeline reproducibility, auditing capabilities, and security against data tampering.

```text
                      Cryptographic Provenance Flow
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXTERNAL DATA ORIGIN (API / Web Portal / Bucket)                            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INGESTION & HASHING LAYER                                                   │
│ - Compute SHA-256 Checksum on Raw Files                                     │
│ - Generate Ingestion Manifest & Timestamps                                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ METADATA & VERSION CONTROL (DVC / LakeFS)                                   │
│ - Commit Dataset Version Pointer                                            │
│ - Register Source URL, License Type, and Lineage Graph in OpenMetadata      │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Cryptographic Verification Formula

$$\text{Digest} = \text{SHA-256}(\text{Raw External Stream}) \implies \text{Immutable Checksum}$$

---

## 7. Automated Ingestion & Quality Audit Architecture

A resilient external data platform uses automated integration patterns to ingest, audit, and normalize incoming external datasets before merging them into feature stores.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA INGESTION ENGINE                       │
├──────────────────────────────┬──────────────────────────────┬───────────────┤
│ Ingestion Layer              │ Legal Compliance Gate        │ Data Versioning│
│ - REST / GraphQL Connectors  │ - License Parser             │ - DVC          │
│ - Web Scrapers (Playwright)  │ - Attribution Generator      │ - LakeFS       │
└──────────────┬───────────────┴──────────────┬───────────────┴───────┬───────┘
               │                              │                       │
               └──────────────────────┬───────┴───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STATISTICAL DRIFT & QUALITY GATE                        │
│ - K-S Test & PSI Engine               - Great Expectations Validation        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CURATED FEATURE STORE PLATFORM                        │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 8. Technology & Integration Matrix

| Functional Area | Industry Standard Tooling | Primary Technical Function |
| --- | --- | --- |
| **Data Versioning** | DVC (Data Version Control), LakeFS | Manages Git-like versioning and immutable snapshots for large external datasets. |
| **Ingestion Frameworks** | Airbyte, Meltano, Apache NiFi | Automated extract-and-load pipeline orchestration across REST APIs and public databases. |
| **Data Quality Validation** | Great Expectations, Soda Core, Deepchecks | Enforces schema validation rules and null-value bounds on incoming third-party files. |
| **Provenance & Metadata** | OpenLineage, Marquez, OpenMetadata | Captures runtime execution lineage and source-to-destination data mappings. |

---

## 9. Personal Understanding

Working through Task 05 underscored that open-source public data offers significant value, but requires systematic compliance and quality auditing before enterprise adoption.
I now see that using external datasets involves more than just downloading files; it requires verifying license restrictions, tracking dataset provenance, and testing for distribution drift.
Without tests like Population Stability Index (PSI) or the Kolmogorov-Smirnov test, external data can introduce hidden covariate shift that degrades downstream machine learning performance.
The core takeaway remains:

> **External data accelerates model capabilities only when ingested through automated, cryptographically traceable pipelines that enforce strict licensing compliance and distribution verification.**

---

## 10. Interview / Viva Questions

### Q1. Why is the CC-BY-NC 4.0 license unsuitable for enterprise production AI systems?

**Answer:**

The CC-BY-NC 4.0 license explicitly prohibits non-commercial usage. Training a commercial machine learning model on NC-licensed data violates those terms and exposes the organization to legal liabilities.

### Q2. How does the Population Stability Index (PSI) help assess external dataset quality?

**Answer:**

PSI measures shifts in variable distributions over time or between datasets. A $\text{PSI} \ge 0.25$ indicates a significant distribution change between the external dataset and baseline internal data, signaling potential dataset drift.

### Q3. What is the difference between Covariate Shift and Concept Shift when integrating public datasets?

**Answer:**

Covariate shift occurs when input feature distributions $P(X)$ change while the conditional probability $P(Y\vert{}X)$ remains constant. Concept shift occurs when the relationship between features and target labels $P(Y\vert{}X)$ changes, even if $P(X)$ stays the same.

### Q4. How does DVC (Data Version Control) track large external datasets without storing raw files in Git?

**Answer:**

DVC creates small text pointer files (`.dvc`) containing cryptographic hashes (e.g., MD5/SHA-256) and file sizes. The raw dataset files are stored in remote object storage (e.g., S3, GCS), while Git tracks only the lightweight pointer files.

### Q5. What is the copyleft hazard associated with datasets governed by the Open Database License (ODbL)?

**Answer:**

ODbL requires that any derivative database or enhanced dataset created using ODbL-licensed source data must also be published under the ODbL. This copyleft requirement can force organizations to open-source proprietary enriched datasets.

### Q6. How does the Kolmogorov-Smirnov (K-S) test identify covariate shift in numerical features?

**Answer:**

The two-sample K-S test compares empirical cumulative distribution functions (ECDFs) of continuous features from external and internal datasets, evaluating whether the maximum distance $D$ between them is statistically significant.

### Q7. Why are SHA-256 checksums calculated upon initial ingestion of external data files?

**Answer:**

SHA-256 hashes generate unique, fixed-length cryptographic fingerprints of raw input files. Recalculating hashes verifies data integrity, detects upstream file modifications, and provides an immutable audit trail.

### Q8. What steps are involved in an automated license compliance audit for external data ingest pipelines?

**Answer:**

1. Extract license metadata from external source endpoints or repository tags.
2. Classify the license against enterprise policy (Permissive, Non-Commercial, Copyleft).
3. Block non-compliant ingestion and automatically generate attribution files for approved permissive sources.

### Q9. What role does Great Expectations play in external data integration pipelines?

**Answer:**

Great Expectations automatically validates incoming datasets against predefined expectations (e.g., non-null bounds, schema types, value ranges), preventing malformed external files from corrupting downstream feature stores.

### Q10. How does a Chi-Square test identify distribution differences in categorical features?

**Answer:**

The Chi-Square test compares observed category frequencies in an external dataset against expected frequencies from an internal baseline. High test statistics indicate significant categorical distribution drift.

### Q11. What is the difference between explicit REST API ingestion and web scraping for external data collection?

**Answer:**

REST APIs provide structured, versioned data contracts with defined rate limits and authentication. Web scraping extracts unstructured HTML content from web pages, making it vulnerable to layout changes and web terms-of-service restrictions.

### Q12. How do OpenLineage and Marquez improve data provenance tracking?

**Answer:**

OpenLineage provides an open standard for capturing metadata events during pipeline runs, while Marquez collects and displays these events to build visual, end-to-end lineage graphs from source endpoints to downstream targets.

### Q13. Why should external feature processing pipelines be isolated from proprietary enterprise code?

**Answer:**

Isolation limits legal exposure by preventing copyleft licenses (e.g., ODbL, GPL) from extending to proprietary internal source code or model weights.

### Q14. What strategies can mitigate API rate limits when ingesting public datasets?

**Answer:**

Pipelines can use exponential backoff, request caching, batch endpoints, and distributed queue systems (e.g., Celery) to manage request volumes within vendor rate limits.

### Q15. How does dataset documentation (e.g., Datasheets for Datasets) improve external dataset evaluation?

**Answer:**

Datasheets detail dataset creation context, collection methodologies, intended usages, potential biases, and legal terms, enabling data teams to evaluate suitability before technical integration.

---

## 11. Conclusion

Task 05 provides an operational framework for ingesting external open-source and third-party datasets into enterprise AI architectures.
The end-to-end data integration flow is summarized below:

```text
External Data Ingestion Pipeline
      ↓
Automated API / Scraping Ingestion
      ↓
License Classification & Legal Compliance Gate
      ↓
Cryptographic Provenance Hashing & DVC Version Control
      ↓
Statistical Drift Testing (K-S Test, PSI, Chi-Square)
      ↓
Curated External Data Ready for Feature Stores

```

The core pillars of external dataset management include:

```text
External Data Architecture Framework
├── Ingestion Engine (APIs, Web Scrapers & Automated Connectors)
├── Compliance & Licensing (Permissive vs. Copyleft & Attribution)
├── Provenance & Immutability (SHA-256 Hashes & DVC Pointers)
└── Quality & Drift Audit (K-S Test, PSI & Schema Validation)

```

Core tools and operational frameworks:

```text
DVC / LakeFS / OpenLineage
Airbyte / Apache NiFi
Great Expectations / Soda Core
SciPy / Statsmodels / Scikit-Learn

```

By enforcing license compliance, tracking cryptographic provenance, and applying statistical quality checks, organizations can safely incorporate external datasets into their enterprise data ecosystem.
The central principle remains:

> **External data accelerates model capabilities only when ingested through automated, cryptographically traceable pipelines that enforce strict licensing compliance and distribution verification.**

---

## 12. Key Takeaways

1. Third-party and open-source public datasets enrich internal machine learning models while reducing annotation costs.
2. Permissive licenses like **CC0**, **MIT**, and **Apache 2.0** allow commercial use with attribution.
3. **CC-BY-NC 4.0** licenses strictly forbid commercial use and cannot be integrated into commercial AI products.
4. Copyleft licenses like **ODbL** require derivative datasets to be shared publicly under identical license terms.
5. **SHA-256 cryptographic hashes** generate immutable fingerprints to verify file integrity during external data ingestion.
6. **DVC (Data Version Control)** manages dataset versioning by storing lightweight pointer files in Git and backing raw assets in object storage.
7. **Population Stability Index (PSI)** quantifies dataset drift, where $\text{PSI} \ge 0.25$ indicates significant distribution shifts.
8. The two-sample **Kolmogorov-Smirnov (K-S) test** compares continuous feature distributions between external and internal datasets to detect covariate shift.
9. **Chi-Square goodness-of-fit tests** evaluate distribution shifts across categorical variables in external datasets.
10. **Great Expectations** automates schema validation, null checks, and value range constraints on incoming third-party files.
11. **OpenLineage** and **Marquez** capture runtime metadata to document source-to-destination data provenance graphs.
12. Isolating external processing layers prevents copyleft license obligations from affecting proprietary internal codebases.
13. Web scraping requires managing dynamic website layouts and complying with site terms of service.
14. Structured APIs offer reliable data contracts, but pipelines must handle rate limits through retry and backoff strategies.
15. **Datasheets for Datasets** document collection context, known biases, and recommended usages for public datasets.
16. Automated license checks prevent non-compliant public data from entering enterprise training pipelines.
17. Combining license verification, cryptographic provenance, and statistical drift auditing ensures reliable use of third-party datasets in production.
