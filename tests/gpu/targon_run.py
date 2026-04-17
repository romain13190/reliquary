"""Deploy Catalyst tests to a Targon H100 and run pytest.

Usage:
    targon run tests/gpu/targon_run.py

What it does:
  1. Builds a pytorch 2.5 + CUDA 12.4 image
  2. Copies the Catalyst source into /app
  3. Installs the reliquary package (editable)
  4. Runs tests/unit tests/integration tests/gpu via pytest
  5. Streams the pytest output back locally as the function result
"""

import targon


IGNORE = [
    "__pycache__",
    ".venv",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "*.pyc",
    "docs/superpowers",
]


image = (
    targon.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "numpy>=1.20.0",
        "transformers>=4.40.0",
        "safetensors>=0.3.0",
        "pydantic>=2.3,<3.0.0",
        "requests>=2.28.0",
        "httpx>=0.25.0",
        "huggingface-hub>=0.20.0",
        "datasets>=2.14.0",
        "aiobotocore>=2.0.0",
        "botocore>=1.24.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "tenacity>=8.0.0",
        "pyarrow>=14.0.0",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
    )
    .pip_install("bittensor>=7.0.0")
    .add_local_dir("/home/ubuntu/Catalyst", "/app", ignore=IGNORE)
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/tmp/hf"})
)


app = targon.App("reliquary-gpu-test", image=image)


@app.function(
    resource=targon.Compute.H100_SMALL,
    min_replicas=0,
    max_replicas=1,
    timeout=1500,
    startup_timeout=900,
)
def run_tests(test_paths: str = "tests/unit tests/integration tests/gpu") -> dict:
    """Install reliquary (editable), then run pytest on the chosen paths."""
    import subprocess
    import sys

    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
        cwd="/app",
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        return {
            "ok": False,
            "stage": "pip-install",
            "returncode": install.returncode,
            "stdout": install.stdout[-4000:],
            "stderr": install.stderr[-4000:],
        }

    import torch
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    paths = test_paths.split()
    pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "--tb=short", "--color=no", *paths],
        cwd="/app",
        capture_output=True,
        text=True,
    )
    return {
        "ok": pytest.returncode == 0,
        "stage": "pytest",
        "returncode": pytest.returncode,
        "gpu": gpu_info,
        "stdout": pytest.stdout[-20000:],
        "stderr": pytest.stderr[-4000:],
    }


@app.local_entrypoint()
async def main(test_paths: str = "tests/unit tests/integration tests/gpu") -> dict:
    print(f"Submitting test run on H100 for paths: {test_paths}")
    result = await run_tests.remote(test_paths=test_paths)
    print()
    print("=" * 80)
    print(f"GPU: {result.get('gpu')}")
    print(f"STAGE: {result.get('stage')}  RC: {result.get('returncode')}  OK: {result.get('ok')}")
    print("=" * 80)
    print("STDOUT (tail):")
    print(result.get("stdout", ""))
    if result.get("stderr"):
        print("=" * 80)
        print("STDERR (tail):")
        print(result["stderr"])
    return result
