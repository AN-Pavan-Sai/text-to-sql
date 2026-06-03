"""
Enterprise Text-to-SQL API

FastAPI application that exposes endpoints for:
  - /retrieve       : Semantic schema retrieval
  - /generate-sql   : LLM-based SQL generation
  - /query          : End-to-end question-to-SQL-to-results pipeline
  - /validate-sql   : SQL syntax validation
  - /benchmark      : Automated evaluation against beaver-query ground truth
  - /health         : Health check
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from config import settings
from retrieval import retriever
from llm_agent import generate_sql
from database import db_manager
from benchmark import run_benchmark

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Enterprise Text-to-SQL API",
    description=(
        "Converts natural-language questions into optimised SQL queries "
        "using semantic schema retrieval and LLM-based generation. "
        "Built on the BeaverBench dataset and evaluated against its "
        "ground-truth annotations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    question: str = Field(..., description="Natural-language question")
    split: str = Field(default="nova", description="Dataset split (dw / nova / neutron)")
    db: Optional[str] = Field(default=None, description="Filter to a specific database name")
    top_k: Optional[int] = Field(default=None, description="Number of tables to retrieve")


class TableDetail(BaseModel):
    relevance_score: float
    reason: str

class RetrieveResponse(BaseModel):
    retrieved_tables: List[str]
    scores: List[float]
    confidence: float
    details: Dict[str, TableDetail]


class GenerateSQLRequest(BaseModel):
    question: str = Field(..., description="Natural-language question")
    use_retrieved_context: bool = Field(default=True, description="Whether to automatically retrieve context")
    schema_context: Optional[str] = Field(default=None, description="Manual schema context if not retrieving")
    split: str = Field(default="nova")
    db: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None)


class GenerateSQLResponse(BaseModel):
    sql: str
    retrieved_tables: List[str]
    is_valid_syntax: bool
    parsing_errors: Optional[str]
    confidence: float
    prompt_used: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural-language question")
    split: str = Field(default="nova", description="Dataset split")
    db: Optional[str] = Field(default=None, description="Database name (auto-detected if omitted)")
    top_k: Optional[int] = Field(default=None, description="Number of tables to retrieve")


class QueryResponse(BaseModel):
    question: str
    retrieved_tables: List[str]
    generated_sql: str
    validation: Dict[str, Any]
    results: Dict[str, Any]
    # Backwards compatibility for cached frontend clients
    is_valid: bool = False
    execution_result: Dict[str, Any] = {}


class ValidateRequest(BaseModel):
    sql: str = Field(..., description="SQL query to validate")
    split: str = Field(default="nova")
    db: str = Field(..., description="Database name to validate against")


class ValidateResponse(BaseModel):
    sql: str
    valid: bool
    error: Optional[str] = None


class BenchmarkResponse(BaseModel):
    total_queries: int
    metrics: Dict[str, float]
    subtask_breakdown: Dict[str, float]
    error_analysis: Dict[str, int]
    sample_details: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    """Pre-load the default dataset split so the first request is fast."""
    logger.info("Pre-loading dataset split: %s", settings.beaver_db_split)
    retriever.load_split(settings.beaver_db_split)
    logger.info("Startup complete.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": settings.groq_model}


@app.post("/retrieve", response_model=RetrieveResponse, tags=["Retrieval"])
async def retrieve_tables(req: RetrieveRequest):
    """
    Retrieve the most relevant tables for a natural-language question.

    Uses semantic embeddings to rank all tables in the specified split
    and returns the top-K results.
    """
    try:
        tables = retriever.retrieve(
            question=req.question,
            split=req.split,
            top_k=req.top_k,
            db_filter=req.db,
        )
        
        retrieved_tables = []
        scores = []
        details = {}
        
        for t in tables:
            name = t["table_name"]
            score = round(t.get("score", 0.0), 2)
            retrieved_tables.append(name)
            scores.append(score)
            details[name] = {
                "relevance_score": score,
                "reason": f"Semantic similarity match for {name}"
            }
            
        confidence = round(sum(scores) / len(scores), 2) if scores else 0.0

        return RetrieveResponse(
            retrieved_tables=retrieved_tables,
            scores=scores,
            confidence=confidence,
            details=details
        )
    except Exception as exc:
        logger.exception("Retrieval error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate-sql", response_model=GenerateSQLResponse, tags=["Generation"])
async def generate_sql_endpoint(req: GenerateSQLRequest):
    """
    Generate SQL from a question using the retrieval + LLM pipeline.
    """
    try:
        tables = []
        schema_ctx = req.schema_context or ""
        scores = []
        
        if req.use_retrieved_context:
            retrieved = retriever.retrieve(
                question=req.question,
                split=req.split,
                top_k=req.top_k,
                db_filter=req.db,
            )
            tables = [t["table_name"] for t in retrieved]
            scores = [t.get("score", 0.0) for t in retrieved]
            
            schema_ctx = retriever.get_schema_context(
                question=req.question,
                split=req.split,
                top_k=req.top_k,
                db_filter=req.db,
            )
            
            # Auto-detect db if not set
            if not req.db and retrieved:
                req.db = retrieved[0]["db"]

        sql, prompt_used = generate_sql(req.question, schema_ctx, return_prompt=True)
        
        # Validation
        db_name = req.db or "nova"
        validation = db_manager.validate_sql(sql, req.split, db_name)
        
        confidence = round(sum(scores) / len(scores), 2) if scores else 0.85
        
        return GenerateSQLResponse(
            sql=sql,
            retrieved_tables=tables,
            is_valid_syntax=validation.get("valid", False),
            parsing_errors=validation.get("error"),
            confidence=confidence,
            prompt_used=prompt_used
        )
    except Exception as exc:
        logger.exception("SQL generation error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query", response_model=QueryResponse, tags=["End-to-End"])
async def query(req: QueryRequest):
    """
    Full end-to-end pipeline: retrieve tables, generate SQL, validate, execute.
    """
    try:
        # Step 1: Retrieve
        tables = retriever.retrieve(
            question=req.question,
            split=req.split,
            top_k=req.top_k,
            db_filter=req.db,
        )
        schema_ctx = retriever.get_schema_context(
            question=req.question,
            split=req.split,
            top_k=req.top_k,
            db_filter=req.db,
        )

        # Determine the database from retrieved tables
        if req.db:
            db_name = req.db
        elif tables:
            db_name = tables[0]["db"]
        else:
            raise HTTPException(
                status_code=400,
                detail="No tables retrieved. Please specify a database name.",
            )

        table_names = [t["table_name"] for t in tables]

        # Step 2: Generate SQL
        sql = generate_sql(req.question, schema_ctx)

        # Step 3: Validate
        validation = db_manager.validate_sql(sql, req.split, db_name)

        # Step 4: Execute
        results = db_manager.execute_sql(sql, req.split, db_name)

        return QueryResponse(
            question=req.question,
            retrieved_tables=table_names,
            generated_sql=sql,
            validation=validation,
            results=results,
            is_valid=validation.get("valid", False),
            execution_result=results
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Query pipeline error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/validate-sql", response_model=ValidateResponse, tags=["Validation"])
async def validate_sql_endpoint(req: ValidateRequest):
    """Validate SQL syntax against a database schema."""
    try:
        result = db_manager.validate_sql(req.sql, req.split, req.db)
        return ValidateResponse(
            sql=req.sql,
            valid=result.get("valid", False),
            error=result.get("error"),
        )
    except Exception as exc:
        logger.exception("Validation error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Benchmark"])
async def benchmark(
    split: str = Query(default=None, description="Dataset split"),
    max_samples: int = Query(default=20, description="Max samples to evaluate"),
    top_k: int = Query(default=None, description="Tables to retrieve per question"),
):
    """
    Run the automated benchmark against beaver-query ground truth.

    Evaluates retrieval recall, SQL validity, exact match, execution match,
    and end-to-end success rate.
    """
    try:
        result = run_benchmark(
            split=split,
            max_samples=max_samples,
            top_k=top_k,
        )
        return BenchmarkResponse(
            total_queries=result["total_queries"],
            metrics=result["metrics"],
            subtask_breakdown=result["subtask_breakdown"],
            error_analysis=result["error_analysis"],
            sample_details=result["sample_details"],
        )
    except Exception as exc:
        logger.exception("Benchmark error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Static files serving has been removed (API-only mode)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
