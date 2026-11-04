"""Cohort checks shared in behavior with the training wrapper."""
import csv
from pathlib import Path

def _cohort_root(destination: Path, feature_subdir: str) -> Path:
    """Accept a cohort directly or inside one directory preserved by download."""
    candidates = [p.parent for p in destination.rglob('labels.csv')
                  if (p.parent / feature_subdir).is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f'{destination}: expected exactly one cohort containing labels.csv and '
            f'{feature_subdir}; found {len(candidates)}. Upload one cohort per artifact.'
        )
    return candidates[0]

def _validate_cohort(root: Path, feature_subdir: str) -> set[int]:
    """Check label format and feature matching before starting training."""
    available = {p.stem for p in (root / feature_subdir).glob('*.h5')}
    with (root / 'labels.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'slide_name', 'slide_label'} <= set(reader.fieldnames or []):
            raise ValueError(f'{root}/labels.csv needs slide_name and slide_label columns')
        rows = list(reader)
    matched = [r for r in rows if r['slide_name'] in available and r['slide_label']]
    if not matched:
        raise ValueError(f'{root}: no labeled slides match the .h5 feature filenames')
    labels = {int(r['slide_label']) for r in matched}
    print(f'{root.name}: {len(matched)} matched labels; {len(rows) - len(matched)} unmatched/empty rows', flush=True)
    return labels
