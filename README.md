# Enterprise Text-to-SQL API

A production-grade FastAPI microservice that converts natural-language questions into optimised SQL queries. The system uses semantic schema retrieval, LLM-based SQL generation (via Groq), and automated benchmarking against the BeaverBench dataset.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Pipeline](#system-pipeline)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Benchmark and Metrics](#benchmark-and-metrics)
- [Design Decisions](#design-decisions)
- [Evaluation Methodology](#evaluation-methodology)

---

## Architecture Overview

The system implements a four-stage pipeline modelled after enterprise BI text-to-SQL tools such as Databricks Assistant and Mode Analytics Query Engine:

1. **Schema Retrieval**: Given a natural-language question, the system uses a sentence-transformer embedding model (`all-MiniLM-L6-v2`) to compute semantic similarity between the question and all available table descriptions. The top-K most relevant tables are returned.

2. **SQL Generation**: The retrieved schema context is combined with the user question and a set of few-shot examples, then sent to the Groq API (using `llama-3.3-70b-versatile`) to generate a syntactically valid and semantically correct SQL query.

3. **Validation and Execution**: The generated SQL is validated against an in-memory SQLite database built from the BeaverBench table schemas and example data. If valid, the query is executed and results are returned.

4. **Benchmarking**: An automated evaluation pipeline runs the full system against the `beaver-query` ground-truth dataset, computing retrieval recall, SQL exact match, execution match, and end-to-end success metrics.

---

## System Pipeline

```
User Question
      |
      v
+---------------------+
| Schema Retrieval    |  sentence-transformers + cosine similarity
| (retrieval.py)      |  -> top-K relevant tables
+---------------------+
      |
      v
+---------------------+
| SQL Generation      |  Groq API (llama-3.3-70b-versatile)
| (llm_agent.py)      |  -> generated SQL query
+---------------------+
      |
      v
+---------------------+
| Validation & Exec   |  SQLite in-memory database
| (database.py)       |  -> syntax check + query results
+---------------------+
      |
      v
+---------------------+
| Benchmark           |  beaver-query ground truth comparison
| (benchmark.py)      |  -> precision, recall, accuracy metrics
+---------------------+
```

---

## Project Structure

```
text-to-sql/
  .env                 # Environment variables (API keys, model config)
  config.py            # Centralised settings (reads from .env)
  main.py              # FastAPI application with all endpoints
  retrieval.py         # Schema retrieval engine (embeddings + similarity)
  llm_agent.py         # LLM SQL generation via Groq API
  database.py          # SQLite database manager (create, validate, execute)
  benchmark.py         # Automated benchmark runner and metrics computation
  requirements.txt     # Python dependencies
  README.md            # This file
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- A Groq API key (set in the `.env` file)

### Steps

1. Clone the repository and navigate to the project directory:

```bash
cd text-to-sql
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify the `.env` file contains the required configuration:

```
GROQ_API_KEY=<your-groq-api-key>
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K_TABLES=6
BEAVER_DB_SPLIT=nova
```

5. Start the development server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

6. Open the interactive API documentation:

```
http://localhost:8000/docs
```

---

## Configuration

All configuration is managed through environment variables loaded from the `.env` file:

| Variable          | Description                                      | Default                    |
|-------------------|--------------------------------------------------|----------------------------|
| GROQ_API_KEY      | Groq API key for LLM access                     | (required)                 |
| GROQ_MODEL        | Groq model identifier                            | llama-3.3-70b-versatile    |
| EMBEDDING_MODEL   | Sentence-transformer model for embeddings        | all-MiniLM-L6-v2           |
| TOP_K_TABLES      | Default number of tables to retrieve             | 6                          |
| BEAVER_DB_SPLIT   | Default BeaverBench split to use                 | nova                       |

---

## API Endpoints

### GET /health

Health check endpoint. Returns the service status and active model name.

**Response:**
```json
{
  "status": "healthy",
  "model": "llama-3.3-70b-versatile"
}
```

### POST /retrieve

Retrieve the most relevant tables for a natural-language question using semantic search.

**Request Body:**
```json
{
  "question": "Show me departments ranked by total enrollment",
  "split": "nova",
  "db": "university",
  "top_k": 5
}
```

**Response:** Returns the matched tables with similarity scores and a formatted schema context string.

### POST /generate-sql

Generate a SQL query from a question and pre-built schema context.

**Request Body:**
```json
{
  "question": "What is the average salary per department?",
  "schema_context": "Table: employees ...\n  Columns: ..."
}
```

**Response:**
```json
{
  "question": "What is the average salary per department?",
  "generated_sql": "SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC;"
}
```

### POST /query

Full end-to-end pipeline: retrieval, generation, validation, and execution in a single call.

**Request Body:**
```json
{
  "question": "Show me departments ranked by total enrollment, excluding online courses",
  "split": "nova",
  "db": "university"
}
```

**Response:** Returns retrieved tables, generated SQL, validation result, and query execution output.

### POST /validate-sql

Validate SQL syntax against a database schema without executing it.

**Request Body:**
```json
{
  "sql": "SELECT * FROM departments;",
  "split": "nova",
  "db": "university"
}
```

### POST /benchmark

Run the automated evaluation pipeline against the beaver-query ground truth.

**Query Parameters:**
- `split` (str, optional): Dataset split to evaluate
- `max_samples` (int, default 20): Number of samples to evaluate
- `top_k` (int, optional): Tables to retrieve per question

**Response:** Returns aggregated metrics and per-sample detail breakdowns.

---

## Benchmark and Metrics

The `/benchmark` endpoint evaluates the system against the BeaverBench ground-truth dataset (`beaver-query`) and reports the following metrics:

| Metric                      | Description                                                              |
|-----------------------------|--------------------------------------------------------------------------|
| retrieval_recall_at_k       | Fraction of ground-truth tables found in the top-K retrieved tables      |
| sql_validity_rate           | Fraction of generated SQL queries that pass syntax validation            |
| sql_exact_match_rate        | Fraction of queries where the normalised SQL matches the reference       |
| sql_execution_match_rate    | Fraction of queries where the result set matches the reference           |
| end_to_end_success_rate     | Fraction of queries that are fully correct across all stages             |

The target for all key metrics is above 0.85 (85%).

---

## Design Decisions

### Why sentence-transformers for retrieval

The `all-MiniLM-L6-v2` model provides a good balance between embedding quality and inference speed. It captures semantic relationships between natural-language questions and technical schema descriptions (table names, column names). This outperforms naive keyword matching, especially for queries that use domain-specific terminology (e.g., "enrollment" mapping to a column called `student_count`).

### Why Groq with llama-3.3-70b-versatile

Groq provides extremely fast inference for open-source LLMs. The `llama-3.3-70b-versatile` model has strong instruction-following capabilities and produces high-quality SQL with few-shot prompting. The combination of a well-structured system prompt, 3 diverse few-shot examples, and the retrieved schema context gives the model enough information to generate correct queries for most enterprise scenarios.

### Why SQLite for validation

SQLite is used for two reasons: (1) it requires zero external infrastructure, making the system self-contained, and (2) the BeaverBench dataset provides example rows that can be loaded into SQLite to verify query execution. The in-memory approach avoids disk I/O overhead.

### Few-shot prompt engineering

The system uses 3 carefully selected few-shot examples covering:
- Multi-table JOINs with filtering (departments/enrollments)
- Simple aggregation (average salary)
- Date filtering with ranking (top customers by spending)

These examples teach the model the expected output format and common SQL patterns without overloading the context window.

---

## Evaluation Methodology

1. **Retrieval evaluation**: For each question in beaver-query, the ground-truth table list is compared against the top-K retrieved tables. Recall is computed as the fraction of ground-truth tables present in the retrieved set.

2. **SQL evaluation**: Generated SQL is compared to the reference SQL in two ways:
   - **Exact match**: Both queries are normalised (lowercased, whitespace-collapsed, semicolons stripped) and compared as strings.
   - **Execution match**: Both queries are executed against the in-memory database and the sorted result sets are compared for equality.

3. **End-to-end evaluation**: A query is considered a full success only if retrieval recall is 1.0 (all ground-truth tables were retrieved) AND the execution results match the reference.

---

## Dataset

This project uses the BeaverBench dataset from HuggingFace:

- Questions and annotations: https://huggingface.co/datasets/beaverbench/beaver-query
- Table schemas: https://huggingface.co/datasets/beaverbench/beaver-table

The datasets are automatically downloaded on first use via the `datasets` library.
