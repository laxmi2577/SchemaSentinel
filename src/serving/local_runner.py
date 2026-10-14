from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict, Field

from src.serving.grammar import (
    extract_query_columns,
    extract_query_tables,
    is_read_only_syntax,
    validate_sql_syntax,
)


class LocalRunnerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: str = "models/SchemaSentinel-7B-Q4_K_M.gguf"
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = 0
    n_batch: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 512
    stop_tokens: List[str] = Field(
        default_factory=lambda: ["<|im_end|>", "```\n", ";\n"]
    )
    enforce_grammar: bool = True
    verbose: bool = False
    dry_run: bool = False


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: str
    raw_text: str
    extracted_sql: str
    tokens_generated: int
    prompt_tokens: int
    latency_seconds: float
    tokens_per_second: float
    is_valid_syntax: bool
    is_read_only: bool
    syntax_error: Optional[str] = None
    tables: List[str] = Field(default_factory=list)
    columns: List[str] = Field(default_factory=list)


def extract_sql_from_response(text: str) -> str:
    cleaned = text.strip()
    fence_pattern = r"```(?:sql)?\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate:
            return candidate

    query_pattern = r"((?:WITH|SELECT)\b[\s\S]*?(?:;|$))"
    query_match = re.search(query_pattern, cleaned, re.IGNORECASE)
    if query_match:
        candidate = query_match.group(1).strip()
        if candidate:
            return candidate

    return cleaned


def format_chatml_sql_prompt(
    schema_ddl: str,
    question: str,
    governance_rules: Optional[Sequence[str]] = None,
) -> str:
    rules = (
        list(governance_rules)
        if governance_rules is not None
        else ["Read-only queries strictly permitted.", "No destructive mutations."]
    )
    rules_text = "\n".join(f"- {r}" for r in rules)

    system_content = (
        "You are SchemaSentinel, a secure text-to-SQL generation engine.\n"
        f"Database Schema:\n{schema_ddl.strip()}\n\n"
        f"Governance Policies:\n{rules_text}\n\n"
        "Generate only compliant ANSI SQL wrapped in ```sql ... ``` fences."
    )

    return (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def check_memory_headroom() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": sys.platform,
        "fits_under_6gb": True,
        "total_ram_gb": 8.0,
        "available_ram_gb": 4.5,
    }
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 2)
        info["available_ram_gb"] = round(mem.available / (1024**3), 2)
        info["fits_under_6gb"] = info["available_ram_gb"] >= 2.0
    except ImportError:
        pass
    return info


class SchemaSentinelCPURunner:
    def __init__(
        self,
        config: Optional[LocalRunnerConfig] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.config = config or LocalRunnerConfig()
        self.llm = llm
        self.init_error: Optional[str] = None
        if self.llm is None and not self.config.dry_run:
            self.llm = self._init_llama_cpp()
        self.is_live = self.llm is not None and not self.config.dry_run

    def _init_llama_cpp(self) -> Optional[Any]:
        if not os.path.exists(self.config.model_path):
            self.init_error = f"Model binary not found at {self.config.model_path}"
            return None
        try:
            from llama_cpp import Llama

            return Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                verbose=self.config.verbose,
            )
        except ImportError:
            self.init_error = "llama-cpp-python is not installed"
            return None
        except Exception as e:
            self.init_error = str(e)
            return None

    def generate_sql_stream(self, prompt: str) -> Iterator[str]:
        if self.llm is None or self.config.dry_run:
            p_lower = prompt.lower()
            if "employees" in p_lower and "tenant" in p_lower:
                tenant_match = re.search(r"tenant\s*(\d+)", p_lower)
                t_id = tenant_match.group(1) if tenant_match else "1"
                if "delete" in p_lower or "drop" in p_lower or "update" in p_lower:
                    query = f"DELETE FROM employees WHERE tenant_id = {t_id} AND salary < 40000;"
                elif "avg" in p_lower or "average" in p_lower:
                    query = f"SELECT department, AVG(salary) AS avg_salary, COUNT(id) AS total_employees FROM employees WHERE tenant_id = {t_id} GROUP BY department;"
                elif "project" in p_lower or "lead" in p_lower:
                    query = f"SELECT p.id, p.budget, e.name AS lead_name FROM projects p JOIN employees e ON p.lead_id = e.id WHERE p.tenant_id = {t_id} AND p.budget > 50000;"
                else:
                    query = f"SELECT id, name, salary FROM employees WHERE tenant_id = {t_id} AND department = 'Engineering' AND salary > 80000;"
                yield "```sql\n"
                yield query + "\n"
                yield "```"
                return

            simulated_chunks = [
                "```sql\n",
                "SELECT ",
                "id, ",
                "name ",
                "FROM ",
                "users ",
                "WHERE ",
                "is_active = 1;\n",
                "```",
            ]
            for chunk in simulated_chunks:
                yield chunk
            return

        stream = self.llm(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=self.config.stop_tokens,
            stream=True,
        )
        for output in stream:
            choices = output.get("choices", [])
            if choices:
                delta = choices[0].get("text", "")
                if delta:
                    yield delta

    def generate_sql(self, prompt: str) -> GenerationResult:
        start_time = time.time()
        chunks: List[str] = []

        for token in self.generate_sql_stream(prompt):
            chunks.append(token)

        latency = time.time() - start_time
        raw_text = "".join(chunks)
        extracted_sql = extract_sql_from_response(raw_text)

        tokens_generated = len(raw_text.split())
        prompt_tokens = len(prompt.split())
        tps = tokens_generated / latency if latency > 0 else 0.0

        is_valid = True
        is_read_only = True
        err_msg: Optional[str] = None
        tables: List[str] = []
        columns: List[str] = []

        if self.config.enforce_grammar and extracted_sql:
            is_valid, err_msg = validate_sql_syntax(extracted_sql)
            is_read_only = is_read_only_syntax(extracted_sql)
            if is_valid:
                tables = extract_query_tables(extracted_sql)
                columns = extract_query_columns(extracted_sql)

        return GenerationResult(
            prompt=prompt,
            raw_text=raw_text,
            extracted_sql=extracted_sql,
            tokens_generated=tokens_generated,
            prompt_tokens=prompt_tokens,
            latency_seconds=latency,
            tokens_per_second=tps,
            is_valid_syntax=is_valid,
            is_read_only=is_read_only,
            syntax_error=err_msg,
            tables=tables,
            columns=columns,
        )

    def generate(
        self,
        prompt: str,
        schema_ddl: Optional[str] = None,
    ) -> GenerationResult:
        return self.generate_sql(prompt=prompt)
