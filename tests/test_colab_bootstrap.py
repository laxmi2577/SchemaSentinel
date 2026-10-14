import importlib
import os
import pytest

sft_module = importlib.import_module("colab.01_qlora_dora_sft")

DriveMountConfig = sft_module.DriveMountConfig
HFAuthConfig = sft_module.HFAuthConfig
WandbConfig = sft_module.WandbConfig
HardwareReport = sft_module.HardwareReport
CloudBootstrapConfig = sft_module.CloudBootstrapConfig
CloudBootstrapStatus = sft_module.CloudBootstrapStatus
audit_gpu_hardware = sft_module.audit_gpu_hardware
mount_google_drive = sft_module.mount_google_drive
setup_huggingface_auth = sft_module.setup_huggingface_auth
init_wandb_session = sft_module.init_wandb_session
ensure_cloud_directories = sft_module.ensure_cloud_directories
bootstrap_cloud_environment = sft_module.bootstrap_cloud_environment
save_adapter_checkpoint = sft_module.save_adapter_checkpoint
run_sft_pipeline = sft_module.run_sft_pipeline
parse_args = sft_module.parse_args


def test_cloud_bootstrap_config_defaults():
    config = CloudBootstrapConfig()
    assert config.drive.mount_point == "/content/drive"
    assert config.drive.target_dir == "/content/drive/MyDrive/SchemaSentinel"
    assert config.hf.token_env_var == "HF_TOKEN"
    assert config.wandb.project == "schemasentinel-sft"
    assert config.wandb.group == "phase3-cloud-sft"
    assert config.base_model_name == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert config.dry_run is False


def test_audit_gpu_hardware():
    report = audit_gpu_hardware()
    assert isinstance(report, HardwareReport)
    assert isinstance(report.cuda_available, bool)
    assert isinstance(report.device_count, int)
    assert isinstance(report.total_memory_gb, float)


def test_mount_google_drive_dry_run_and_non_colab():
    assert mount_google_drive(dry_run=True) is True
    assert mount_google_drive(dry_run=False) is False


def test_setup_huggingface_auth_logic(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert setup_huggingface_auth(dry_run=True) is True
    assert setup_huggingface_auth(dry_run=False) is False

    assert setup_huggingface_auth(explicit_token="hf_test_mock_token", dry_run=False) is True
    assert os.environ.get("HF_TOKEN") == "hf_test_mock_token"


def test_ensure_cloud_directories(tmp_path):
    target_dir = str(tmp_path / "SchemaSentinel")
    checkpoints_dir = str(tmp_path / "SchemaSentinel" / "checkpoints" / "sft")
    logs_dir = str(tmp_path / "SchemaSentinel" / "logs")
    datasets_dir = str(tmp_path / "SchemaSentinel" / "datasets")

    cfg = DriveMountConfig(
        target_dir=target_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        datasets_dir=datasets_dir,
    )

    created = ensure_cloud_directories(cfg, dry_run=False)
    assert len(created) == 4
    for p in created:
        assert os.path.exists(p)


def test_bootstrap_cloud_environment_dry_run():
    config = CloudBootstrapConfig(dry_run=True)
    status = bootstrap_cloud_environment(config)
    assert isinstance(status, CloudBootstrapStatus)
    assert status.drive_mounted is True
    assert status.drive_path == "/content/drive/MyDrive/SchemaSentinel"
    assert status.hf_authenticated is True
    assert status.wandb_initialized is True
    assert len(status.directories_created) == 4


def test_cli_argument_parsing():
    args = parse_args(["--dry-run", "--mount-point", "/tmp/drive", "--wandb-project", "custom-sft"])
    assert args.dry_run is True
    assert args.mount_point == "/tmp/drive"
    assert args.wandb_project == "custom-sft"


def test_save_adapter_checkpoint(tmp_path):
    out_dir = str(tmp_path / "checkpoints")
    ckpt_path = save_adapter_checkpoint(
        model=None,
        tokenizer=None,
        output_dir=out_dir,
        step=50,
        is_final=False,
        metrics={"loss": 0.35},
    )
    assert os.path.exists(ckpt_path)
    state_file = os.path.join(ckpt_path, "trainer_state.json")
    assert os.path.exists(state_file)

    import json

    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["step"] == 50
    assert data["is_final"] is False
    assert data["metrics"]["loss"] == 0.35


def test_run_sft_pipeline_dry_run(tmp_path):
    out_dir = str(tmp_path / "drive" / "checkpoints")
    config = CloudBootstrapConfig(
        drive=DriveMountConfig(checkpoints_dir=out_dir),
        dry_run=True,
    )
    result = run_sft_pipeline(config=config, dry_run=True)
    assert result["status"] == "dry_run_completed"
    assert result["adapter_type"] == "dora"
    assert "checkpoint_path" in result
    assert os.path.exists(result["checkpoint_path"])


def test_cli_argument_parsing_extended():
    args = parse_args([
        "--dry-run",
        "--adapter-type",
        "qlora",
        "--learning-rate",
        "1e-4",
        "--num-epochs",
        "5",
        "--lambda-ul",
        "0.75",
    ])
    assert args.adapter_type == "qlora"
    assert args.learning_rate == 1e-4
    assert args.num_epochs == 5
    assert args.lambda_ul == 0.75
