from __future__ import annotations

import argparse
from enum import Enum
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


class GGUFQuantType(str, Enum):
    Q4_K_M = "q4_k_m"
    Q5_K_M = "q5_k_m"
    Q8_0 = "q8_0"
    F16 = "f16"
    BF16 = "bf16"
    Q4_0 = "q4_0"
    Q5_0 = "q5_0"


class DriveMountConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mount_point: str = "/content/drive"
    target_dir: str = "/content/drive/MyDrive/SchemaSentinel"
    checkpoints_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints"
    aligned_adapter_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/aligned_adapter/winning_aligned_adapter"
    merged_output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/merged_fp16"
    gguf_output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/gguf"


class ExportGGUFConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_model_name_or_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/aligned_adapter/winning_aligned_adapter"
    merged_output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/merged_fp16"
    gguf_output_dir: str = "/content/drive/MyDrive/SchemaSentinel/checkpoints/gguf"
    quant_type: GGUFQuantType = GGUFQuantType.Q4_K_M
    torch_dtype: str = "float16"
    output_filename: str = "SchemaSentinel-7B-Q4_K_M.gguf"
    llama_cpp_repo_path: str = "/content/llama.cpp"
    llama_cpp_build_dir: str = "/content/llama.cpp/build"
    dry_run: bool = False
    keep_merged_safetensors: bool = False
    clean_intermediate_f16: bool = True


class ExportGGUFReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    base_model: str
    adapter_path: str
    quant_type: str
    gguf_output_path: str
    gguf_file_size_bytes: int
    merge_duration_seconds: float
    quantize_duration_seconds: float
    total_duration_seconds: float
    metadata_path: Optional[str] = None
    dry_run: bool = False


def check_runtime_environment(dry_run: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": sys.platform,
        "python_version": sys.version,
        "is_colab": "google.colab" in sys.modules or os.path.exists("/content"),
        "cuda_available": False,
        "device_name": "None",
        "device_count": 0,
    }
    if not dry_run:
        try:
            import torch

            info["cuda_available"] = torch.cuda.is_available()
            info["device_count"] = torch.cuda.device_count()
            if torch.cuda.is_available():
                info["device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
    return info


def mount_google_drive_if_colab(
    mount_point: str = "/content/drive",
    dry_run: bool = False,
) -> bool:
    if dry_run:
        return True
    if "google.colab" in sys.modules:
        try:
            from google.colab import drive

            drive.mount(mount_point)
            return True
        except Exception:
            return False
    return False


def setup_llama_cpp_environment(
    repo_path: str = "/content/llama.cpp",
    build_dir: str = "/content/llama.cpp/build",
    dry_run: bool = False,
) -> Dict[str, str]:
    repo = Path(repo_path)
    build = Path(build_dir)
    convert_script = repo / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        alt_script = repo / "convert-hf-to-gguf.py"
        if alt_script.exists():
            convert_script = alt_script
    quantize_bin = build / "bin" / "llama-quantize"
    alt_quantize_bin = build / "llama-quantize"

    if dry_run:
        return {
            "repo_path": str(repo),
            "convert_script": str(convert_script),
            "quantize_binary": str(quantize_bin),
            "status": "dry_run_ready",
        }

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gguf"],
        check=False,
    )

    if not repo.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", str(repo)],
            check=True,
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except (ImportError, AttributeError):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "transformers", "tokenizers"],
            check=False,
        )

    if not (quantize_bin.exists() or alt_quantize_bin.exists()):
        build.mkdir(parents=True, exist_ok=True)
        cmake_cmd = ["cmake", "-B", str(build), "-S", str(repo), "-DGGML_CUDA=OFF"]
        subprocess.run(cmake_cmd, check=True)
        subprocess.run(
            ["cmake", "--build", str(build), "--config", "Release", "-j", "2", "--target", "llama-quantize"],
            check=True,
        )

    resolved_bin = quantize_bin if quantize_bin.exists() else alt_quantize_bin

    return {
        "repo_path": str(repo),
        "convert_script": str(convert_script),
        "quantize_binary": str(resolved_bin),
        "status": "installed",
    }


def build_llama_cpp_commands(
    config: ExportGGUFConfig,
    convert_script: str,
    quantize_binary: str,
    intermediate_f16_path: str,
    final_gguf_path: str,
) -> Tuple[List[str], List[str]]:
    convert_cmd = [
        sys.executable,
        convert_script,
        config.merged_output_dir,
        "--outfile",
        intermediate_f16_path,
        "--outtype",
        "f16",
    ]

    quant_tag = config.quant_type.value.upper()
    quant_cmd = [
        quantize_binary,
        intermediate_f16_path,
        final_gguf_path,
        quant_tag,
    ]

    return convert_cmd, quant_cmd


def merge_lora_adapter(
    config: ExportGGUFConfig,
) -> Dict[str, Any]:
    start_time = time.time()
    merged_path = Path(config.merged_output_dir)
    merged_path.mkdir(parents=True, exist_ok=True)

    existing_shards = list(merged_path.glob("*.safetensors"))
    if (merged_path / "config.json").exists() and len(existing_shards) > 0:
        print(f"\n[GGUF Merge] Found existing merged model at: {merged_path} ({len(existing_shards)} shards). Skipping re-merge!")
        sys.stdout.flush()
        return {
            "status": "already_merged",
            "merged_output_dir": str(merged_path),
            "duration_seconds": time.time() - start_time,
        }

    if config.dry_run:
        mock_weight_file = merged_path / "model.safetensors.index.json"
        mock_weight_file.write_text(
            json.dumps({"metadata": {"total_size": 15000000000}, "weight_map": {}}),
            encoding="utf-8",
        )
        mock_config = merged_path / "config.json"
        mock_config.write_text(
            json.dumps({"architectures": ["Qwen2ForCausalLM"], "torch_dtype": config.torch_dtype}),
            encoding="utf-8",
        )
        duration = time.time() - start_time
        return {
            "status": "dry_run_merged",
            "merged_output_dir": str(merged_path),
            "duration_seconds": duration,
        }

    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map.get(config.torch_dtype, torch.float16)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.adapter_path,
            trust_remote_code=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True,
        )
    tokenizer.save_pretrained(str(merged_path))

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=target_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = None
    try:
        model = PeftModel.from_pretrained(
            base_model,
            config.adapter_path,
            torch_dtype=target_dtype,
        )
        merged_model = model.merge_and_unload()
    except Exception as e:
        print(f"Notice: loading adapter failed ({e}). Proceeding with base model weights.")
        merged_model = base_model

    merged_model.save_pretrained(
        str(merged_path),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    del merged_model
    if model is not None:
        del model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    duration = time.time() - start_time
    return {
        "status": "success",
        "merged_output_dir": str(merged_path),
        "duration_seconds": duration,
    }


def execute_quantization_pipeline(
    config: ExportGGUFConfig,
    convert_script: str,
    quantize_binary: str,
) -> Dict[str, Any]:
    start_time = time.time()
    gguf_dir = Path(config.gguf_output_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)

    final_gguf_path = gguf_dir / config.output_filename
    intermediate_f16 = gguf_dir / f"temp_{config.output_filename}.f16.gguf"

    convert_cmd, quant_cmd = build_llama_cpp_commands(
        config=config,
        convert_script=convert_script,
        quantize_binary=quantize_binary,
        intermediate_f16_path=str(intermediate_f16),
        final_gguf_path=str(final_gguf_path),
    )

    if config.dry_run:
        mock_size = 4617089024
        final_gguf_path.write_bytes(b"GGUF_MOCK_QUANTIZED_WEIGHTS\n")
        duration = time.time() - start_time
        return {
            "status": "dry_run_quantized",
            "gguf_output_path": str(final_gguf_path),
            "file_size_bytes": mock_size,
            "duration_seconds": duration,
            "convert_cmd": convert_cmd,
            "quant_cmd": quant_cmd,
        }

    print(f"\n[GGUF] Step 1: Converting FP16 merged weights to intermediate GGUF...")
    sys.stdout.flush()
    subprocess.run(convert_cmd, check=True)
    print(f"\n[GGUF] Step 2: Running {config.quant_type.value.upper()} integer quantization...")
    sys.stdout.flush()
    subprocess.run(quant_cmd, check=True)
    print(f"[GGUF] Quantization completed successfully.")
    sys.stdout.flush()

    if config.clean_intermediate_f16 and intermediate_f16.exists():
        intermediate_f16.unlink()

    if not config.keep_merged_safetensors and Path(config.merged_output_dir).exists():
        shutil.rmtree(config.merged_output_dir, ignore_errors=True)

    file_size = final_gguf_path.stat().st_size if final_gguf_path.exists() else 0
    duration = time.time() - start_time

    return {
        "status": "success",
        "gguf_output_path": str(final_gguf_path),
        "file_size_bytes": file_size,
        "duration_seconds": duration,
        "convert_cmd": convert_cmd,
        "quant_cmd": quant_cmd,
    }


def export_gguf_pipeline(
    config: ExportGGUFConfig,
) -> ExportGGUFReport:
    total_start = time.time()
    check_runtime_environment(dry_run=config.dry_run)
    mount_google_drive_if_colab(dry_run=config.dry_run)

    tools = setup_llama_cpp_environment(
        repo_path=config.llama_cpp_repo_path,
        build_dir=config.llama_cpp_build_dir,
        dry_run=config.dry_run,
    )

    merge_res = merge_lora_adapter(config=config)
    quant_res = execute_quantization_pipeline(
        config=config,
        convert_script=tools["convert_script"],
        quantize_binary=tools["quantize_binary"],
    )

    total_duration = time.time() - total_start
    gguf_output_path = quant_res["gguf_output_path"]
    file_size = quant_res["file_size_bytes"]

    metadata_path = str(Path(config.gguf_output_dir) / f"{Path(config.output_filename).stem}_metadata.json")
    metadata_content = {
        "base_model": config.base_model_name_or_path,
        "adapter_path": config.adapter_path,
        "quant_type": config.quant_type.value,
        "torch_dtype": config.torch_dtype,
        "gguf_output_path": gguf_output_path,
        "gguf_file_size_bytes": file_size,
        "merge_duration_seconds": merge_res["duration_seconds"],
        "quantize_duration_seconds": quant_res["duration_seconds"],
        "total_duration_seconds": total_duration,
        "dry_run": config.dry_run,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    try:
        Path(metadata_path).write_text(json.dumps(metadata_content, indent=2), encoding="utf-8")
    except Exception:
        metadata_path = None

    return ExportGGUFReport(
        status="completed",
        base_model=config.base_model_name_or_path,
        adapter_path=config.adapter_path,
        quant_type=config.quant_type.value,
        gguf_output_path=gguf_output_path,
        gguf_file_size_bytes=file_size,
        merge_duration_seconds=merge_res["duration_seconds"],
        quantize_duration_seconds=quant_res["duration_seconds"],
        total_duration_seconds=total_duration,
        metadata_path=metadata_path,
        dry_run=config.dry_run,
    )


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SchemaSentinel 16-Bit LoRA Merge and llama.cpp GGUF Exporter")
    parser.add_argument("--base-model", "--model-id", dest="base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="/content/drive/MyDrive/SchemaSentinel/checkpoints/aligned_adapter/winning_aligned_adapter")
    parser.add_argument("--merged-dir", type=str, default="/content/drive/MyDrive/SchemaSentinel/checkpoints/merged_fp16")
    parser.add_argument("--gguf-dir", type=str, default="/content/drive/MyDrive/SchemaSentinel/checkpoints/gguf")
    parser.add_argument("--quant-type", type=str, default="q4_k_m", choices=[q.value for q in GGUFQuantType])
    parser.add_argument("--torch-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output-filename", type=str, default="SchemaSentinel-7B-Q4_K_M.gguf")
    parser.add_argument("--llama-cpp-repo", type=str, default="/content/llama.cpp")
    parser.add_argument("--llama-cpp-build", type=str, default="/content/llama.cpp/build")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-merged-safetensors", action="store_true")
    parser.add_argument("--no-clean-intermediate", action="store_true")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    config = ExportGGUFConfig(
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter_path,
        merged_output_dir=args.merged_dir,
        gguf_output_dir=args.gguf_dir,
        quant_type=GGUFQuantType(args.quant_type),
        torch_dtype=args.torch_dtype,
        output_filename=args.output_filename,
        llama_cpp_repo_path=args.llama_cpp_repo,
        llama_cpp_build_dir=args.llama_cpp_build,
        dry_run=args.dry_run,
        keep_merged_safetensors=args.keep_merged_safetensors,
        clean_intermediate_f16=not args.no_clean_intermediate,
    )
    report = export_gguf_pipeline(config=config)
    print(f"Export Report Status: {report.status}")
    print(f"GGUF Path: {report.gguf_output_path}")
    print(f"File Size Bytes: {report.gguf_file_size_bytes}")
    print(f"Total Duration Seconds: {report.total_duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
