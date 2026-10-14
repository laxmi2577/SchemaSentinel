import importlib
import json
from pathlib import Path
import pytest

export_module = importlib.import_module("colab.03_export_gguf")

GGUFQuantType = export_module.GGUFQuantType
DriveMountConfig = export_module.DriveMountConfig
ExportGGUFConfig = export_module.ExportGGUFConfig
ExportGGUFReport = export_module.ExportGGUFReport
check_runtime_environment = export_module.check_runtime_environment
mount_google_drive_if_colab = export_module.mount_google_drive_if_colab
setup_llama_cpp_environment = export_module.setup_llama_cpp_environment
build_llama_cpp_commands = export_module.build_llama_cpp_commands
merge_lora_adapter = export_module.merge_lora_adapter
execute_quantization_pipeline = export_module.execute_quantization_pipeline
export_gguf_pipeline = export_module.export_gguf_pipeline
parse_args = export_module.parse_args


def test_gguf_quant_type_enum():
    assert GGUFQuantType.Q4_K_M.value == "q4_k_m"
    assert GGUFQuantType.Q5_K_M.value == "q5_k_m"
    assert GGUFQuantType.Q8_0.value == "q8_0"
    assert GGUFQuantType.F16.value == "f16"
    assert GGUFQuantType.BF16.value == "bf16"


def test_export_gguf_config_defaults():
    config = ExportGGUFConfig()
    assert config.base_model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert "aligned_adapter" in config.adapter_path
    assert config.quant_type == GGUFQuantType.Q4_K_M
    assert config.torch_dtype == "float16"
    assert config.output_filename == "SchemaSentinel-7B-Q4_K_M.gguf"
    assert config.dry_run is False
    assert config.clean_intermediate_f16 is True


def test_drive_mount_config():
    mount_cfg = DriveMountConfig()
    assert mount_cfg.mount_point == "/content/drive"
    assert "SchemaSentinel" in mount_cfg.target_dir
    assert "checkpoints" in mount_cfg.checkpoints_dir
    assert "gguf" in mount_cfg.gguf_output_dir


def test_check_runtime_environment():
    info = check_runtime_environment(dry_run=True)
    assert "platform" in info
    assert "python_version" in info
    assert "is_colab" in info
    assert info["cuda_available"] is False


def test_mount_google_drive_dry_run():
    mounted = mount_google_drive_if_colab(dry_run=True)
    assert mounted is True


def test_setup_llama_cpp_environment_dry_run(tmp_path: Path):
    repo = tmp_path / "llama.cpp"
    build = tmp_path / "llama.cpp" / "build"
    tools = setup_llama_cpp_environment(
        repo_path=str(repo),
        build_dir=str(build),
        dry_run=True,
    )
    assert tools["status"] == "dry_run_ready"
    assert "convert_hf_to_gguf.py" in tools["convert_script"]
    assert "llama-quantize" in tools["quantize_binary"]


def test_build_llama_cpp_commands(tmp_path: Path):
    config = ExportGGUFConfig(
        merged_output_dir=str(tmp_path / "merged"),
        gguf_output_dir=str(tmp_path / "gguf"),
        quant_type=GGUFQuantType.Q4_K_M,
        output_filename="test.gguf",
    )
    convert_script = str(tmp_path / "convert.py")
    quantize_bin = str(tmp_path / "quantize")
    intermediate = str(tmp_path / "intermediate.f16.gguf")
    final_path = str(tmp_path / "test.gguf")

    convert_cmd, quant_cmd = build_llama_cpp_commands(
        config=config,
        convert_script=convert_script,
        quantize_binary=quantize_bin,
        intermediate_f16_path=intermediate,
        final_gguf_path=final_path,
    )

    assert convert_cmd[1] == convert_script
    assert convert_cmd[2] == str(tmp_path / "merged")
    assert convert_cmd[4] == intermediate
    assert convert_cmd[6] == "f16"

    assert quant_cmd[0] == quantize_bin
    assert quant_cmd[1] == intermediate
    assert quant_cmd[2] == final_path
    assert quant_cmd[3] == "Q4_K_M"


def test_merge_lora_adapter_dry_run(tmp_path: Path):
    merged_dir = tmp_path / "merged_weights"
    config = ExportGGUFConfig(
        merged_output_dir=str(merged_dir),
        dry_run=True,
    )

    result = merge_lora_adapter(config)
    assert result["status"] == "dry_run_merged"
    assert result["duration_seconds"] >= 0.0
    assert (merged_dir / "config.json").exists()
    assert (merged_dir / "model.safetensors.index.json").exists()


def test_execute_quantization_pipeline_dry_run(tmp_path: Path):
    gguf_dir = tmp_path / "gguf_out"
    config = ExportGGUFConfig(
        gguf_output_dir=str(gguf_dir),
        output_filename="model_q4_k_m.gguf",
        dry_run=True,
    )

    result = execute_quantization_pipeline(
        config=config,
        convert_script=str(tmp_path / "convert.py"),
        quantize_binary=str(tmp_path / "quantize"),
    )

    assert result["status"] == "dry_run_quantized"
    assert result["file_size_bytes"] > 1000000
    assert Path(result["gguf_output_path"]).exists()
    assert len(result["convert_cmd"]) > 0
    assert len(result["quant_cmd"]) > 0


def test_export_gguf_pipeline_dry_run_end_to_end(tmp_path: Path):
    merged_dir = tmp_path / "merged_fp16"
    gguf_dir = tmp_path / "gguf_final"

    config = ExportGGUFConfig(
        base_model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path=str(tmp_path / "mock_adapter"),
        merged_output_dir=str(merged_dir),
        gguf_output_dir=str(gguf_dir),
        quant_type=GGUFQuantType.Q4_K_M,
        output_filename="SchemaSentinel-7B-Q4_K_M.gguf",
        dry_run=True,
    )

    report = export_gguf_pipeline(config)

    assert isinstance(report, ExportGGUFReport)
    assert report.status == "completed"
    assert report.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert report.quant_type == "q4_k_m"
    assert report.dry_run is True
    assert Path(report.gguf_output_path).exists()
    assert report.metadata_path is not None
    assert Path(report.metadata_path).exists()

    metadata_bytes = Path(report.metadata_path).read_text(encoding="utf-8")
    metadata = json.loads(metadata_bytes)
    assert metadata["base_model"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert metadata["quant_type"] == "q4_k_m"
    assert metadata["dry_run"] is True


def test_parse_args_cli():
    cli_args = [
        "--base-model", "Qwen/Qwen2.5-Coder-7B-Instruct",
        "--quant-type", "q5_k_m",
        "--torch-dtype", "bfloat16",
        "--output-filename", "custom_7b.gguf",
        "--dry-run",
        "--keep-merged-safetensors",
    ]
    parsed = parse_args(cli_args)
    assert parsed.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert parsed.quant_type == "q5_k_m"
    assert parsed.torch_dtype == "bfloat16"
    assert parsed.output_filename == "custom_7b.gguf"
    assert parsed.dry_run is True
    assert parsed.keep_merged_safetensors is True
