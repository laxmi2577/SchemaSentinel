import pytest
from src.serving.local_runner import (
    LocalRunnerConfig,
    GenerationResult,
    extract_sql_from_response,
    format_chatml_sql_prompt,
    check_memory_headroom,
    SchemaSentinelCPURunner,
)


def test_local_runner_config_defaults():
    config = LocalRunnerConfig()
    assert "SchemaSentinel-7B-Q4_K_M.gguf" in config.model_path
    assert config.n_ctx == 4096
    assert config.n_threads == 4
    assert config.n_gpu_layers == 0
    assert config.n_batch == 512
    assert config.temperature == 0.0
    assert config.top_p == 0.9
    assert config.max_tokens == 512
    assert "<|im_end|>" in config.stop_tokens
    assert config.enforce_grammar is True
    assert config.dry_run is False


def test_extract_sql_from_response_fenced():
    fenced_1 = "Here is the query:\n```sql\nSELECT id, name FROM users WHERE id = 10;\n```\nExplanation follows."
    sql_1 = extract_sql_from_response(fenced_1)
    assert sql_1 == "SELECT id, name FROM users WHERE id = 10;"

    fenced_2 = "```\nWITH cte AS (SELECT 1) SELECT * FROM cte;\n```"
    sql_2 = extract_sql_from_response(fenced_2)
    assert sql_2 == "WITH cte AS (SELECT 1) SELECT * FROM cte;"


def test_extract_sql_from_response_unfenced():
    raw_query = "The requested answer is SELECT region, SUM(amount) FROM sales GROUP BY region; Hope this helps."
    extracted = extract_sql_from_response(raw_query)
    assert "SELECT region, SUM(amount) FROM sales GROUP BY region;" in extracted


def test_format_chatml_sql_prompt():
    ddl = "CREATE TABLE employees (id INT, department TEXT, salary FLOAT);"
    question = "Which employees earn over 80000?"
    rules = ["Only SELECT queries permitted.", "Tenant isolation required."]

    prompt = format_chatml_sql_prompt(schema_ddl=ddl, question=question, governance_rules=rules)
    assert "<|im_start|>system" in prompt
    assert ddl in prompt
    assert "Tenant isolation required." in prompt
    assert "<|im_start|>user\nWhich employees earn over 80000?<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_check_memory_headroom():
    info = check_memory_headroom()
    assert "platform" in info
    assert "fits_under_6gb" in info
    assert "total_ram_gb" in info
    assert "available_ram_gb" in info
    assert isinstance(info["fits_under_6gb"], bool)


def test_cpu_runner_simulated_stream():
    runner = SchemaSentinelCPURunner(config=LocalRunnerConfig(dry_run=True))
    chunks = list(runner.generate_sql_stream("Find active users"))
    full_text = "".join(chunks)

    assert len(chunks) > 0
    assert "SELECT" in full_text
    assert "FROM users" in full_text


def test_cpu_runner_generate_sql_success():
    runner = SchemaSentinelCPURunner(config=LocalRunnerConfig(dry_run=True, enforce_grammar=True))
    result = runner.generate_sql("Retrieve all active users")

    assert isinstance(result, GenerationResult)
    assert result.is_valid_syntax is True
    assert result.is_read_only is True
    assert result.syntax_error is None
    assert "users" in result.tables
    assert "id" in result.columns
    assert "name" in result.columns
    assert result.latency_seconds >= 0.0
    assert result.tokens_generated > 0


def test_cpu_runner_with_mock_llm():
    def mock_llm_call(prompt, **kwargs):
        tokens = [
            {"choices": [{"text": "```sql\n"}]},
            {"choices": [{"text": "SELECT customer_id, total FROM orders WHERE total > 100;\n"}]},
            {"choices": [{"text": "```"}]},
        ]
        for t in tokens:
            yield t

    runner = SchemaSentinelCPURunner(
        config=LocalRunnerConfig(dry_run=False, enforce_grammar=True),
        llm=mock_llm_call,
    )

    result = runner.generate_sql("Show orders over 100")
    assert result.is_valid_syntax is True
    assert result.is_read_only is True
    assert "orders" in result.tables
    assert "total" in result.columns
    assert "customer_id" in result.columns


def test_cpu_runner_grammar_enforcement_malformed():
    def mock_malformed_llm(prompt, **kwargs):
        tokens = [
            {"choices": [{"text": "```sql\n"}]},
            {"choices": [{"text": "DROP TABLE critical_data;\n"}]},
            {"choices": [{"text": "```"}]},
        ]
        for t in tokens:
            yield t

    runner = SchemaSentinelCPURunner(
        config=LocalRunnerConfig(dry_run=False, enforce_grammar=True),
        llm=mock_malformed_llm,
    )

    result = runner.generate_sql("Execute command")
    assert result.is_valid_syntax is False
    assert result.is_read_only is False
    assert result.syntax_error is not None
    assert "Prohibited mutation keyword" in result.syntax_error
