"""DigitalHub entrypoint for the final, CPU-only statistical analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def analyze_statistics(
    project, training_results, collection, fm_shift, layer_shift, top64_shift,
    output_artifact_name="4-statistics", include_illustration=False,
):
    """Download completed step 2/3 artifacts and publish tables and figures.

    Each input may contain a single output directory or multiple output
    directories bundled into one artifact for a multi-model experiment.
    """
    base = Path("local_data").resolve()
    base.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="statistics-", dir=base))
    inputs = dict(training_results=training_results, collection=collection,
                  fm_shift=fm_shift, layer_shift=layer_shift, top64_shift=top64_shift)
    paths = {}
    for name, artifact in inputs.items():
        destination = work / name
        destination.mkdir()
        print(f"Downloading {name}: {artifact.key}", flush=True)
        artifact.download(str(destination), overwrite=True)
        paths[name] = str(destination)
    output = work / "reports"
    output.mkdir()
    request = work / "request.json"
    request.write_text(json.dumps(dict(paths=paths, output=str(output),
                                      include_illustration=include_illustration)))
    subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("report.py")),
                    str(request)], check=True)
    expected = ["training_runs_with_shift_metrics.csv", "analysis_status.json", "README.md"]
    if any(not (output / name).is_file() for name in expected):
        raise RuntimeError("Statistics did not produce the required outputs")
    (output / "digitalhub_inputs.json").write_text(json.dumps({
        "inputs": {name: artifact.key for name, artifact in inputs.items()},
        "settings": {"include_illustration": include_illustration},
        "upstream_commit": "c85d68e87237c9075dde9694c92803944d8a085f",
    }, indent=2))
    project.log_artifact(name=output_artifact_name, kind="artifact", source=str(output))
