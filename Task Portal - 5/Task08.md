# Task 08 — Big Data Processing Frameworks, Distributed Feature Extraction & Real-Time Streaming Data Pipelines

## 1. Task Information

| Field | Details |
|---|---|
| Internship | Data Science Internship — Portal V |
| Task Number | 08 |
| Topic | Big Data Engineering — Distributed Processing (PySpark, Dask, Ray), Streaming Pipelines (Kafka, Flink), Schema Evolution & Distributed Feature Extraction |
| Task Type | Technical Core & Enterprise Big Data Engineering |
| Status | Completed |
| Repository Section | `tasks/portal-05/task-08/` |

---

## 2. Objective

The objective of this task is to design, implement, and benchmark high-throughput, enterprise-scale **Distributed Big Data Processing Engines & Streaming Analytics Pipelines** capable of executing feature extraction, data transformations, and real-time model scoring over multi-terabyte datasets.
This task focuses on:
- Comparative evaluation of distributed execution frameworks: PySpark (Resilient Distributed Datasets, DataFrames, Catalyst Optimizer), Dask (Task Graphs), and Ray Core/Data (Actor/Task Paradigm for Distributed ML).
- Designing fault-tolerant real-time event streaming architectures using Apache Kafka, Apache Flink, and Spark Structured Streaming.
- Implementing distributed feature engineering at scale (Tf-Idf, Rolling Window Aggregations, Vector Assemblers, Sparse Vector Encodings).
- Managing enterprise schema evolution, data contracts, and serialization protocols (Apache Avro, Protocol Buffers, Parquet, Delta Lake).
- Benchmarking distributed execution topologies, partition strategies, memory tuning (Executor Heap, Off-Heap, Garbage Collection), and data shuffling bottlenecks.

---

## 3. Introduction

Big data engineering forms the computational backbone of modern enterprise AI. As incoming data volumes scale from gigabytes to petabytes, single-node data frames (e.g., Pandas) fail due to Out-Of-Memory (OOM) errors and CPU saturation.

```text
                     Distributed Big Data Ecosystem
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Unstructured &   │ ───► │ Distributed      │ ───► │ Distributed      │
│ Streaming Inputs │      │ Storage (S3/Delta│      │ Processing Engine│
└──────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                             │
┌──────────────────┐      ┌──────────────────┐               │
│ Real-Time Model  │ ◄─── │ Low-Latency Event│ ◄─────────────┘
│ Inference & Lakes│      │ Stream (Flink)   │
└──────────────────┘      └──────────────────┘

```

Distributed computing resolves memory constraints by partitioning data and parallelizing execution across multi-node clusters. However, distributed pipelines introduce challenges such as data skew, high network shuffle costs, partition imbalances, and out-of-order event streams.
The core operating principle for big data processing is:

> **Distributed processing scales linearly only when computational workflows minimize data shuffling across nodes, optimize partition boundaries, and stream data asynchronously.**

---

## 4. Distributed Processing Engines: Architectural Comparison

Selecting the appropriate processing engine depends on computational workloads, node memory limits, and integration targets.

```text
                   Distributed Processing Framework Comparison
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ Execution Framework                   │ Key Architectural Mechanics           │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Apache Spark (PySpark)                │ DAG-based execution driven by Catalyst│
│                                       │ Query Optimizer and Tungsten Memory   │
│                                       │ Engine; optimal for large-scale ETL.  │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Dask                                  │ Dynamic task graphs scaled via Native │
│                                       │ Python primitives; ideal for Pandas/  │
│                                       │ NumPy parallelization.                │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Ray Core & Ray Data                   │ Actor-based, low-latency task         │
│                                       │ scheduling with zero-copy shared      │
│                                       │ memory (Plasma Store) for AI models.  │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ Apache Flink                          │ True stateful event-at-a-time streaming│
│                                       │ engine with low-latency checkpointing │
│                                       │ and exact-once semantics.             │
└───────────────────────────────────────┴───────────────────────────────────────┘

```

---

## 5. Mathematical & Algorithmic Foundations

Distributed computing relies on mathematical partition models, windowing functions, and streaming state mechanics to execute computations in parallel.

### 5.1 Distributed Partition Scaling & Shuffle Complexity

Given a dataset of size $N$ partitioned across $P$ nodes, a map-side transformation operates in parallel local time:

$$T_{\text{map}} = \mathcal{O}\left( \frac{N}{P} \right)$$

When executing a wide transformation (e.g., `groupByKey`, `JOIN`), data must be redistributed across all cluster nodes. The network shuffle communication complexity scales quadratically with respect to partition exchanges:

$$T_{\text{shuffle}} = \mathcal{O}\left( N \log N + \frac{P^2 \cdot M_{\text{record}}}{B_{\text{network}}} \right)$$

Where $M_{\text{record}}$ is record byte size, and $B_{\text{network}}$ is cluster network bandwidth. Minimizing wide shuffles via **Salting Keys** or **Broadcast Joins** optimizes performance.

---

### 5.2 Event-Time Sliding Windowing & Watermarking Logic

In streaming engines (Flink, Spark Streaming), real-time aggregation over out-of-order events relies on event timestamps $t_E$ and bounded out-of-orderness watermarks $W(t)$:

$$W(t) = \max(t_E) - \Delta t_{\text{allowed\_lateness}}$$

An event occurring at time $t_E$ is processed within window $[T_{\text{start}}, T_{\text{end}})$ if:

$$T_{\text{start}} \le t_E < T_{\text{end}} \quad \text{and} \quad t_E \ge W(t)$$

Events arriving with $t_E < W(t)$ are flagged as late records and routed to side outputs or dropped to preserve state bounds.

```text
                     Watermark Event Processing Timeline
Event Flow:    ───(t=10)─────(t=18)─────(t=12 [Late])─────(t=25)───►
Watermark W(t):                        └── W(t) = 18 - 5 = 13
                                           │
Action:                                    ▼ (t=12 < W(t)=13 ──► Dropped/Side Output)

```

---

### 5.3 Scaled Distributed TF-IDF Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) converts unstructured textual logs into sparse feature vectors across distributed nodes:

$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$

$$\text{IDF}(t, D) = \ln \left( \frac{\vert{}D\vert{}}{1 + \vert{}\{d \in D : t \in d\}\vert{}} \right)$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

In PySpark ML, the `HashingTF` transformer applies the feature hashing trick to map terms to feature indices via a uniform hash function, eliminating global dictionary synchronization bottlenecks across worker nodes.

---

## 6. Real-Time Streaming Architecture & Schema Evolution

High-throughput event-driven systems require resilient streaming platforms coupled with strict schema validation protocols.

```text
                     Real-Time Streaming Pipeline Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ EVENT GENERATORS (IoT Telemetry, App Clickstream, Log Aggregators)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ APACHE KAFKA EVENT INGESTION BUS                                            │
│ - Multi-Topic Partitions with Key-Based Routing                             │
│ - Schema Registry (Confluent / Glue) enforcing Avro / Protobuf Contracts     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STREAM PROCESSING ENGINE (Apache Flink / Spark Structured Streaming)        │
│ - Stateful Event-Time Sliding Window Aggregations                           │
│ - Low-Latency Checkpointing & Watermark Out-of-Order Handling               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────┴──────────────────────────────────────┐
│                                                                             │
▼                                                                             ▼
┌──────────────────────────────────────┐     ┌────────────────────────────────┐
│ REAL-TIME FEATURE STORE (Redis)      │     │ ANALYTICAL LAKEHOUSE           │
│ - Serving online low-latency ML      │     │ - Delta Lake / Apache Iceberg  │
└──────────────────────────────────────┘     └────────────────────────────────┘

```

---

## 7. Distributed System Tuning & Partition Optimization Framework

Optimizing big data clusters requires balancing executor memory, managing GC overhead, and preventing partition skew.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED PERFORMANCE OPTIMIZATION ENGINE               │
├──────────────────────────────┬──────────────────────────────┬───────────────┤
│ Memory Architecture          │ Partition Management         │ Join Optimization│
│ - Executor Heap vs. Off-Heap │ - Re-partition vs. Coalesce  │ - Broadcast   │
│ - G1GC Garbage Collection    │ - Salt Keys for Skew Mitigation│ - Salting   │
└──────────────┬───────────────┴──────────────┬───────────────┴───────┬───────┘
               │                              │                       │
               └──────────────────────┬───────┴───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OPTIMIZED HIGH-THROUGHPUT CLUSTER                     │
└─────────────────────────────────────────────────────────────────────────────┘

```

### Partitioning & Memory Optimization Guidelines

* **Broadcast Hash Join Threshold:** When joining a large dataset ($>1\text{TB}$) with a small lookup table ($<100\text{MB}$), use broadcast joins (`broadcast(small_df)`) to bypass full shuffle phases.
* **Partition Sizing:** Aim for target partition sizes between $100\text{MB}$ and $200\text{MB}$ uncompressed in memory. Avoid over-partitioning, which creates small-file bottlenecks.
* **Salting Skewed Keys:** To eliminate executor bottlenecks caused by highly skewed keys (e.g., millions of events sharing one ID), append a random salt integer ($1 \dots K$) to distribute work across worker threads.

---

## 8. Technology & Integration Matrix

| Functional Area | Industry Standard Tooling | Primary Operational Function |
| --- | --- | --- |
| **Batch Processing Engines** | Apache Spark (PySpark), Ray Data, Dask | Executes distributed parallel transformations, map-reduce tasks, and feature extraction. |
| **Real-Time Stream Engines** | Apache Flink, Spark Structured Streaming | Processes low-latency windowed aggregations and stateful event transformations. |
| **Event Ingestion & Messaging** | Apache Kafka, Apache Pulsar | Distributed event log bus providing partitioned, fault-tolerant message streaming. |
| **Schema Governance** | Confluent Schema Registry, Apache Avro, Protobuf | Enforces data serialization contracts, preventing pipeline breaks during upstream changes. |

---

## 9. Personal Understanding

Task 08 highlighted that distributed computing requires careful optimization of data placement, memory allocation, and network usage.
I now see that scaling PySpark or Flink jobs involves managing execution plans, data shuffle bottlenecks, and partition imbalances, rather than simply adding cluster nodes.
Using strategy patterns like broadcast joins, key salting, and event-time watermarking helps build real-time processing pipelines that remain stable under heavy data loads.
The core takeaway remains:

> **Distributed processing scales linearly only when computational workflows minimize data shuffling across nodes, optimize partition boundaries, and stream data asynchronously.**

---

## 10. Interview / Viva Questions

### Q1. What is the fundamental difference between `repartition()` and `coalesce()` in Apache Spark?

**Answer:**

`repartition()` performs a full network shuffle to redistribute data evenly into a specified number of partitions (increasing or decreasing partition counts). `coalesce()` reduces the number of partitions by merging adjacent local partitions on the same node, avoiding a full network shuffle.

### Q2. How does Spark's Catalyst Optimizer optimize physical execution plans?

**Answer:**

Catalyst applies rule-based and cost-based optimizations to logical query plans. It reorders operations, pushes down predicates to file readers (predicate pushdown), prunes unused columns, and selects optimal physical join strategies (e.g., Broadcast Hash Join).

### Q3. What is a Broadcast Hash Join in PySpark, and when should it be used?

**Answer:**

A Broadcast Hash Join copies a small DataFrame to all worker nodes in a cluster, building an in-memory hash table locally. This avoids a costly network shuffle of the larger dataset. It is ideal when joining a large dataset with a small dimension table (typically $<100\text{MB}$).

### Q4. How do watermarks handle out-of-order events in Apache Flink and Spark Structured Streaming?

**Answer:**

Watermarks track event-time progress by defining a threshold ($\Delta t$) for acceptable event lateness. Any event with an event timestamp older than the current watermark is treated as late data and either dropped or routed to side outputs, maintaining state size boundaries.

### Q5. What is the cause of "data skew" in distributed computing, and how can it be mitigated using key salting?

**Answer:**

Data skew occurs when a small subset of partition keys contains a disproportionately large volume of records, overwhelming individual executor nodes. Key salting appends random integers to the key, spreading records evenly across multiple worker nodes during join or aggregation operations.

### Q6. How does Ray Data achieve zero-copy memory reads across distributed tasks?

**Answer:**

Ray uses an in-memory object store (Plasma Store) on each node. Shared object references can be read concurrently by multiple local worker processes without memory copying or serialization overhead.

### Q7. What are the key operational differences between Apache Avro and Apache Parquet?

**Answer:**

Apache Avro is a row-based binary serialization format optimized for fast, write-heavy streaming ingestion with Schema Registry support. Apache Parquet is a columnar storage format optimized for read-heavy analytical queries, column pruning, and compression.

### Q8. What is the function of the Confluent Schema Registry in Kafka pipelines?

**Answer:**

The Schema Registry serves as a centralized metadata repository that stores Avro/Protobuf schemas. It verifies schema compatibility (e.g., backward, forward) for incoming Kafka messages before writes, preventing broken data pipelines.

### Q9. How does the Tungsten execution engine improve Spark memory efficiency?

**Answer:**

Tungsten uses off-heap memory management with binary row encodings to bypass Java GC overhead, leverages cache-aware algorithm designs, and generates dynamic bytecode at runtime to optimize CPU register utilization.

### Q10. What is the difference between tumbling windows and sliding windows in real-time streaming?

**Answer:**

Tumbling windows are fixed-size, non-overlapping time intervals (e.g., every 5 minutes). Sliding windows have a fixed duration but overlap based on a slide interval (e.g., a 10-minute window that updates every 1 minute).

### Q11. What causes Java Garbage Collection (GC) pauses in PySpark executors, and how can they be minimized?

**Answer:**

Long GC pauses occur when millions of small transient Java objects accumulate in executor heap space. Switching to G1GC, tuning `spark.memory.fraction`, and shifting memory storage off-heap mitigates GC pauses.

### Q12. How does Dask's dynamic task graph execution differ from Spark's execution model?

**Answer:**

Spark operates primarily on structured DataFrames/RDDs with bulk-synchronous parallel stages. Dask builds dynamic, low-overhead task graphs using arbitrary Python code, native NumPy arrays, and Pandas DataFrames, making it flexible for custom scientific computing.

### Q13. What is predicate pushdown, and why is it important for big data storage engines?

**Answer:**

Predicate pushdown evaluates SQL `WHERE` filter conditions directly at the storage layer (e.g., reading Parquet file footers) before loading data into memory. This drastically reduces network I/O and memory overhead by reading only relevant data blocks.

### Q14. What are exact-once processing semantics in stream processing, and how are they achieved in Flink?

**Answer:**

Exactly-once semantics guarantee that each incoming event influences state end-to-end exactly once, even during node failures. Flink achieves this by combining lightweight state checkpointing (Asynchronous Barrier Snapshotting) with two-phase commit protocol sinks.

### Q15. Why is HashingTF faster than CountVectorizer in distributed PySpark ML pipelines?

**Answer:**

`HashingTF` uses the feature hashing trick to map terms to numeric indices using a deterministic hash function without building a global vocabulary index. `CountVectorizer` requires a global pass over all distributed data to construct a master vocabulary, requiring extra network synchronization.

---

## 11. Conclusion

Task 08 outlines a comprehensive architecture for running distributed data processing and real-time streaming operations at scale.
The complete data engineering workflow is summarized below:

```text
Distributed Data Engineering Architecture
      ↓
Real-Time & Batch Ingestion (Kafka + Schema Registry)
      ↓
Distributed Streaming Engine (Flink / Spark Structured Streaming)
      ↓
Distributed Transformation & Feature Extraction (PySpark / Ray)
      ↓
Optimized Cluster Execution (Broadcast Joins, Salted Keys, Off-Heap Memory)
      ↓
Scalable Downstream Lakehouse & Online Feature Store

```

The core pillars of distributed big data processing include:

```text
Big Data Engineering Framework
├── Processing Engines (PySpark, Dask, Ray Data, Apache Flink)
├── Streaming Infrastructure (Apache Kafka, Event-Time Watermarking)
├── Feature Extraction at Scale (HashingTF, Sparse Vector Encodings)
└── Performance Optimization (Broadcast Joins, Key Salting, Off-Heap Tuning)

```

Core tools and operational frameworks:

```text
Apache Spark (PySpark) / Ray / Dask
Apache Flink / Spark Structured Streaming
Apache Kafka / Confluent Schema Registry
Delta Lake / Apache Parquet / Apache Avro

```

By leveraging distributed compute frameworks, optimizing network shuffle costs, and enforcing schema governance, data engineering teams can build resilient, low-latency processing systems for enterprise AI.
The central principle remains:

> **Distributed processing scales linearly only when computational workflows minimize data shuffling across nodes, optimize partition boundaries, and stream data asynchronously.**

---

## 12. Key Takeaways

1. Distributed data computing resolves single-node out-of-memory errors by partitioning operations across compute clusters.
2. **PySpark** uses the Catalyst Optimizer and Tungsten Memory Engine for large-scale batch ETL workloads.
3. **Ray Data** provides low-latency, actor-based dynamic task scheduling with zero-copy shared memory for distributed ML workloads.
4. Network data shuffling ($P^2$ complexity) is a primary bottleneck in distributed execution; minimizing wide transformations improves throughput.
5. **Broadcast Hash Joins** bypass costly shuffles by copying small tables ($<100\text{MB}$) to all worker nodes.
6. **Key Salting** mitigates data skew by appending random integers to high-cardinality keys, spreading workloads evenly across nodes.
7. **Apache Flink** provides stateful event-at-a-time processing with low-latency checkpointing and exact-once delivery semantics.
8. **Watermarking** manages out-of-order events in streaming pipelines by establishing late-arrival thresholds relative to event time.
9. **Confluent Schema Registry** enforces Avro/Protobuf serialization contracts on Kafka streams, preventing pipeline breaks.
10. **HashingTF** uses feature hashing to convert text into sparse vectors in parallel without global dictionary synchronizations.
11. `coalesce()` reduces partition counts locally without triggering a full network shuffle, unlike `repartition()`.
12. **Predicate pushdown** evaluates filter conditions directly at the Parquet/Delta storage layer to avoid reading unnecessary data blocks into memory.
13. Off-heap memory allocation bypasses Java GC pauses during high-volume distributed data processing.
14. Target individual partition sizes between $100\text{MB}$ and $200\text{MB}$ in memory to balance cluster parallelism and file I/O overhead.
15. **Dask** parallelizes native Pandas and NumPy workflows using dynamic task execution graphs.
16. Exactly-once processing in Flink combines state checkpointing with two-phase commit sinks to ensure data consistency during node failures.
17. Combining distributed processing engines, stream management, and schema governance creates a strong foundation for scalable enterprise AI data systems.
