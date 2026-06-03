"""
LLM-based SQL generation via the Groq API.

Constructs a carefully engineered prompt containing the user question,
retrieved schema context, and few-shot examples, then calls the Groq
chat completions endpoint to produce a SQL query.
"""

import re
import logging
from typing import Optional

from groq import Groq

from config import settings

logger = logging.getLogger(__name__)

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL analyst. Given a natural language question and the relevant database schema, generate a single, correct, optimised SQL query that answers the question.

Rules:
1. Output ONLY the SQL query. Do not include any explanation, markdown formatting, or code fences.
2. Use standard SQL syntax compatible with SQLite.
3. Use proper JOINs when multiple tables are needed.
4. Use aliases for readability.
5. Handle NULLs appropriately.
6. Use aggregation functions (COUNT, SUM, AVG, etc.) where the question implies them.
7. If the question asks for a ranking or ordering, include ORDER BY and optionally LIMIT.
8. Do not use SELECT * unless absolutely necessary; select only the required columns.
9. Ensure the query is syntactically valid and would execute without errors.
10. When filtering, use appropriate WHERE clauses based on the question context."""

FEW_SHOT_EXAMPLES = """
Example 1:
Schema:
  Table: departments (database: university)
    Columns: dept_id (INTEGER), dept_name (TEXT), building (TEXT)
  Table: courses (database: university)
    Columns: course_id (INTEGER), title (TEXT), dept_id (INTEGER), is_online (BOOLEAN)
  Table: enrollments (database: university)
    Columns: enrollment_id (INTEGER), student_id (INTEGER), course_id (INTEGER), semester (TEXT)

Question: Show me departments ranked by total enrollment, excluding online courses.
SQL: SELECT d.dept_name, COUNT(e.enrollment_id) AS total_enrollment FROM departments d JOIN courses c ON d.dept_id = c.dept_id JOIN enrollments e ON c.course_id = e.course_id WHERE c.is_online = 0 GROUP BY d.dept_name ORDER BY total_enrollment DESC

Example 2:
Schema:
  Table: employees (database: company)
    Columns: emp_id (INTEGER), name (TEXT), department (TEXT), salary (REAL), hire_date (DATE)

Question: What is the average salary per department?
SQL: SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC

Example 3:
Schema:
  Table: orders (database: ecommerce)
    Columns: order_id (INTEGER), customer_id (INTEGER), total (REAL), order_date (DATE), status (TEXT)
  Table: customers (database: ecommerce)
    Columns: customer_id (INTEGER), name (TEXT), region (TEXT)

Question: Find the top 5 customers by total spending in 2024.
SQL: SELECT c.name, SUM(o.total) AS total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE strftime('%Y', o.order_date) = '2024' GROUP BY c.customer_id, c.name ORDER BY total_spent DESC LIMIT 5
"""


def generate_sql(
    question: str,
    schema_context: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Generate a SQL query from a natural-language question.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    schema_context : str
        A formatted string describing the relevant tables and columns,
        typically produced by ``SchemaRetriever.get_schema_context``.
    temperature : float
        Sampling temperature (0 = deterministic).
    max_tokens : int
        Maximum tokens in the response.

    Returns
    -------
    str
        The generated SQL query string.
    """
    user_message = (
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Now answer the following:\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {question}\n"
        f"SQL:"
    )

    client = _get_client()

    logger.info("Calling Groq API model=%s for question: %s", settings.groq_model, question[:80])

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stream=False,
    )

    raw = response.choices[0].message.content.strip()
    sql = _clean_sql(raw)
    logger.info("Generated SQL: %s", sql[:200])
    return sql


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and extraneous text from LLM output."""
    # Remove ```sql ... ``` wrappers
    raw = re.sub(r"^```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    # If multiple statements, keep only the first
    lines = raw.strip().split(";")
    sql = lines[0].strip()
    if sql and not sql.endswith(";"):
        sql += ";"

    return sql
