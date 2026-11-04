"""DigitalHub wrapper for TRIDENT whole-slide feature extraction."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


def _normalise_gpu_ids(gpus: Iterable[int] | int | str | None) -> list[int]:
    """Return GPU IDs in the form expected by TRIDENT's ``--gpus`` option."""
    if gpus is None:
        return [0, 1]
    if isinstance(gpus, int):
        return [gpus]
    if isinstance(gpus, str):
        values = gpus.replace(",", " ").split()
    else:
        values = list(gpus)

    gpu_ids = [int(value) for value in values]
    if not gpu_ids:
        raise ValueError("gpus must contain at least one GPU ID")
    return gpu_ids


def _copy_labels_csv(download_dir: Path, job_dir: Path) -> Path:
    """Place the downloaded labels at the cohort root as ``labels.csv``."""
    csv_files = sorted(path for path in download_dir.rglob("*.csv") if path.is_file())
    preferred = [path for path in csv_files if path.name.lower() == "labels.csv"]

    if preferred:
        source = preferred[0]
    elif len(csv_files) == 1:
        source = csv_files[0]
    elif not csv_files:
        raise FileNotFoundError("The labels artifact does not contain a CSV file")
    else:
        names = ", ".join(str(path.relative_to(download_dir)) for path in csv_files)
        raise ValueError(
            "The labels artifact contains multiple CSV files and none is named labels.csv: "
            f"{names}"
        )

    destination = job_dir / "labels.csv"
    shutil.copy2(source, destination)
    return destination


def _prepare_conch_v15_cache(environment: dict[str, str]) -> None:
    """Authenticate once and cache CONCH weights before GPU workers start."""
    token = environment.get("HF_TOKEN") or environment.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "CONCH v1.5 requires a Hugging Face token. Create a DigitalHub secret "
            "named HF_TOKEN and include secrets=['HF_TOKEN'] in the job run."
        )

    # Hugging Face Hub reads HF_TOKEN in the worker processes. Explicitly use
    # the same token and cache here so two TRIDENT GPU workers do not race to
    # download the gated checkpoint independently.
    environment["HF_TOKEN"] = token
    cache_dir = Path(environment["HF_HOME"]) / "hub"

    try:
        from huggingface_hub import hf_hub_download

        for filename in ("meta.yaml", "pytorch_model_vision.bin"):
            hf_hub_download(
                repo_id="MahmoodLab/conchv1_5",
                filename=filename,
                token=token,
                cache_dir=str(cache_dir),
            )
    except Exception as exc:
        raise RuntimeError(
            "Unable to access the gated MahmoodLab/conchv1_5 repository. "
            "Confirm that the Hugging Face account owning HF_TOKEN has been "
            "granted model access and that the token has read permission."
        ) from exc


def trident_feature_extraction(
    project,
    slides,
    labels,
    output_artifact_name: str,
    task: str = "all",
    patch_encoder: str = "conch_v15",
    mag: float = 20,
    patch_size: int = 512,
    gpus: Iterable[int] | int | str | None = None,
    slide_encoder: str | None = None,
):
    """Run TRIDENT on a slides artifact and log the generated cohort directory.

    ``project`` is injected by the DigitalHub Python runtime. ``slides`` and
    ``labels`` are DigitalHub artifacts supplied through ``Function.run(inputs=...)``.
    """
    local_data_dir = Path("local_data")
    local_data_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="trident-", dir=local_data_dir))
    wsi_dir = work_dir / "wsis"
    labels_dir = work_dir / "labels"
    job_dir = work_dir / "train"

    for directory in (wsi_dir, labels_dir, job_dir):
        directory.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.setdefault("TRIDENT_HOME", str((work_dir / "trident-cache").resolve()))
    environment.setdefault("HF_HOME", str((work_dir / "huggingface-cache").resolve()))

    if task in {"all", "feat"} and (slide_encoder == "titan" or
                                            (slide_encoder is None and patch_encoder == "conch_v15")):
        print("Checking access to CONCH v1.5 and preparing its model cache")
        _prepare_conch_v15_cache(environment)

    print(f"Downloading slides to {wsi_dir}")
    slides.download(str(wsi_dir), overwrite=True)
    if not any(path.is_file() for path in wsi_dir.rglob("*")):
        raise FileNotFoundError("The slides artifact did not produce any files")

    print(f"Downloading labels to {labels_dir}")
    labels.download(str(labels_dir), overwrite=True)
    labels_path = _copy_labels_csv(labels_dir, job_dir)
    print(f"Using labels file {labels_path}")

    gpu_ids = _normalise_gpu_ids(gpus)
    batch_script = Path(__file__).with_name("run_batch_of_slides.py")
    command = [
        sys.executable,
        str(batch_script),
        "--task",
        task,
        "--wsi_dir",
        str(wsi_dir),
        "--job_dir",
        str(job_dir),
        "--patch_encoder",
        patch_encoder,
        "--mag",
        str(mag),
        "--patch_size",
        str(patch_size),
        "--gpus",
        *(str(gpu_id) for gpu_id in gpu_ids),
    ]
    if slide_encoder is not None:
        command.extend(["--slide_encoder", slide_encoder])

    print(f"Running: {shlex.join(command)}")
    subprocess.run(command, check=True, env=environment)

    print(f"Logging TRIDENT outputs from {job_dir}")
    project.log_artifact(
        name=output_artifact_name,
        kind="artifact",
        source=str(job_dir),
    )
