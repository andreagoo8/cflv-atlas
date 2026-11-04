import importlib.util
import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report import build_reports, prediction_summary


def fixture_inputs(tmp_path, multiple=False):
    paths = {name: tmp_path / name for name in
             ["training_results", "collection", "fm_shift", "layer_shift", "top64_shift"]}
    for path in paths.values():
        path.mkdir(parents=True)
    models = ["conch_v15", "virchow2", "uni_v2"] if multiple else ["conch_v15"]
    collected, fm_rows, layer_rows, top_rows = [], [], [], []
    for f, fm in enumerate(models):
        subdir = f"20x_512px_0px_overlap/features_{fm}"
        for r in range(3 if multiple else 1):
            name = f"custom_{fm}_{r}"  # Intentionally has no upstream naming convention.
            run = paths["training_results"] / name
            run.mkdir()
            cfg = dict(run_name=name, mode="patch", tile_pooling="abmil", feature_subdir=subdir,
                       num_patches=2000 + r * 1000, n_heads=4, head_dim=256, hidden_dim=512,
                       dropout=0.2, no_gated=False, lr=3e-4, weight_decay=1e-4)
            (run / "config.json").write_text(json.dumps(cfg))
            pd.DataFrame({"slide_name": ["a", "b"], "true_label": [0, 1], "pred_label": [0, 1],
                          "prob_class_0": [.8, .3], "prob_class_1": [.2, .7]}).to_csv(
                run / "preds_internal_test.csv", index=False)
            pd.DataFrame({"slide_name": ["c", "d", "x_sessile_serrated"],
                          "true_label": [0, 1, 0], "pred_label": [0, 1, 0],
                          "prob_class_0": [.6, .4, 1.0], "prob_class_1": [.4, .6, 0.0]}).to_csv(
                run / "preds_external_test.csv", index=False)
            collected.append(dict(run_name=name, mode="patch", tile_pooling="abmil",
                                  feature_subdir=subdir, internal_test_mcc=.9 - .02*r,
                                  external_test_no_ssl_mcc=.8 - .03*r - .02*f))
            for rep in range(3):
                sizes = dict(n_internal_total=100, n_external_no_ssl_total=90,
                             n_internal_sample=80, n_external_sample=80)
                metrics = dict(mmd_raw=.1*(f+1)+.01*rep+.003*r, fid_raw=f+rep+.2*r+1,
                               coral_raw=.01*(r+f+rep+1), energy_raw=.2*(f+1)+.01*r,
                               knn_domain_mixing=.8-.02*(f+r+rep))
                drop = .05*(f+1)+.015*(f+1)*r+.001*rep
                common = dict(run_name=name, feature_subdir=subdir, repetition=rep,
                              mcc_internal=.9, mcc_external=.9-drop, mcc_drop=drop,
                              **sizes, **metrics)
                for layer in ["pre_attention_mean", "post_attention", "penultimate"]:
                    layer_rows.append(dict(common, layer_name=layer))
                top_rows.append(dict(common, layer_name="topk64_positive_evidence"))
                if r == 0:
                    fm_rows.append(dict(fm_name=fm, feature_subdir=subdir, repetition=rep,
                                        **sizes, **metrics))
    pd.DataFrame(collected).to_csv(paths['collection'] / 'training_runs.csv', index=False)
    for name, rows, analysis in [('fm_shift', fm_rows, 'fm'), ('layer_shift', layer_rows, 'layers'),
                                  ('top64_shift', top_rows, 'top64')]:
        pd.DataFrame(rows).to_csv(paths[name] / 'repeats.csv', index=False)
        manifest = dict(inputs=dict(internal_features='internal:1', external_features='external:1'),
                        settings=dict(analysis=analysis, exclude_external_ssl=True, sample_external=True,
                                      feature_subdirs=sorted({r['feature_subdir'] for r in rows}),
                                      command_options={key: value for key, value in
                                                       [('seed', '1234'), ('n-repeats', '3'),
                                                        ('target-per-domain', '80'), ('knn-k', '15'),
                                                        ('sigma-sample-size', '512'), ('energy-sample-size', '2048'),
                                                        ('energy-chunk-size', '512'), ('eps', '1e-12')]}))
        (paths[name] / 'digitalhub_inputs.json').write_text(json.dumps(manifest))
    return paths


def test_enrichment_values_and_ssl_exclusion(tmp_path):
    path = tmp_path / 'predictions.csv'
    pd.DataFrame({'slide_name': ['a', 'b', 'x_sessile_serrated'],
                  'prob_class_0': [1., .5, .9], 'prob_class_1': [0., .5, .1]}).to_csv(path, index=False)
    actual = prediction_summary(path, True)
    assert actual['entropy'] == pytest.approx(np.log(2)/2)
    assert actual['confidence'] == pytest.approx(.75)
    assert actual['margin'] == pytest.approx(.5)
    assert actual['n'] == 2
    assert prediction_summary(path, False)['n'] == 3
    data = pd.read_csv(path)
    data.loc[0, 'prob_class_0'] = 2
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match='invalid class probabilities'):
        prediction_summary(path, False)


def test_single_run_end_to_end(tmp_path):
    paths = fixture_inputs(tmp_path)
    out = tmp_path / 'output'
    build_reports(paths, out)
    result = pd.read_csv(out / 'training_runs_with_shift_metrics.csv')
    assert result.confidence_drop.iloc[0] == pytest.approx(.15)
    assert result.margin_drop.iloc[0] == pytest.approx(.3)
    assert result.entropy_shift.iloc[0] > 0
    status = json.loads((out/'analysis_status.json').read_text())
    assert {x['cell'] for x in status if x.get('cell') and x['status'] == 'skipped'} == {4, 7, 8, 9, 10}
    table = pd.read_csv(out/'figures/panel2_shift_metric_correlation_table_with_raw_fm.csv')
    assert len(table) == 25
    assert table.median_spearman_rho.isna().all()
    assert (out/'figures/panel1B_mcc_drop_by_fm.pdf').is_file()
    assert not list(out.glob('figures/illustration_*'))


def test_single_class_predictions_explain_recovery(tmp_path):
    path = tmp_path / 'preds_internal_test.csv'
    pd.DataFrame({'slide_name': ['a', 'b'], 'true_label': [0, 0],
                  'pred_label': [0, 0], 'prob_class_0': [1., 1.]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='train with at least two classes'):
        prediction_summary(path, False)


def test_multiple_fms_runs_and_repeats(tmp_path):
    paths = fixture_inputs(tmp_path, multiple=True)
    collection_path = paths['collection']/'training_runs.csv'
    collected = pd.read_csv(collection_path)
    source = paths['training_results']/collected.run_name.iloc[0]
    for pooling in ['mean', 'max']:
        name = f'baseline_{pooling}'
        dest = paths['training_results']/name
        shutil.copytree(source, dest)
        cfg = json.loads((dest/'config.json').read_text())
        cfg.update(run_name=name, tile_pooling=pooling)
        (dest/'config.json').write_text(json.dumps(cfg))
        row = collected.iloc[0].to_dict()
        row.update(run_name=name, tile_pooling=pooling)
        collected = pd.concat([collected, pd.DataFrame([row])], ignore_index=True)
    collected.to_csv(collection_path, index=False)
    out = tmp_path/'output'
    build_reports(paths, out, include_illustration=True)
    status = json.loads((out/'analysis_status.json').read_text())
    assert all(x['status'] == 'completed' for x in status)
    assert (out/'figures/illustration_representation_shift.svg').is_file()
    instability = pd.read_csv(out/'figures/panel3A_fm_repetition_mcc_drop_instability.csv')
    assert len(instability) == 9
    # SD of [a, a+.015, a+.030] for CONCH; layer rows must not duplicate runs.
    assert instability.loc[instability.fm == 'Conch v1.5', 'mcc_drop_instability'].tolist() == pytest.approx([.015]*3)
    assert instability.n_runs.eq(3).all()
    correlations = pd.read_csv(out/'figures/panel2_shift_metric_repeat_correlations_with_raw_fm.csv')
    assert correlations.spearman_rho.notna().any()
    assert (out/'figures/supp_abmil_hyperparameter_global_tests.csv').is_file()
    methods = pd.read_csv(out/'figures/supp_method_comparison_abmil_average_max_summary.csv')
    assert set(methods.method) == {'ABMIL', 'Average', 'Max'}


@pytest.mark.parametrize('change', ['seed', 'ssl', 'cohort', 'duplicate', 'mcc', 'missing_repeat'])
def test_reject_incompatible_reports(tmp_path, change):
    paths = fixture_inputs(tmp_path)
    manifest = paths['top64_shift']/'digitalhub_inputs.json'
    data = json.loads(manifest.read_text())
    if change == 'seed':
        data['settings']['command_options']['seed'] = '42'
    elif change == 'ssl':
        data['settings']['exclude_external_ssl'] = False
    elif change == 'cohort':
        data['inputs']['external_features'] = 'different:1'
    elif change == 'duplicate':
        folder = paths['top64_shift']/'duplicate'
        folder.mkdir()
        shutil.copy(paths['top64_shift']/'repeats.csv', folder/'repeats.csv')
    elif change == 'mcc':
        csv = paths['top64_shift']/'repeats.csv'
        frame = pd.read_csv(csv)
        frame.loc[0, 'mcc_drop'] = .9
        frame.to_csv(csv, index=False)
    elif change == 'missing_repeat':
        csv = paths['top64_shift']/'repeats.csv'
        pd.read_csv(csv).iloc[:-1].to_csv(csv, index=False)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build_reports(paths, tmp_path/'output')


def test_digitalhub_handler_downloads_and_publishes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('statistics_handler', ROOT/'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = fixture_inputs(tmp_path/'inputs')
    class Artifact:
        def __init__(self, path):
            self.path, self.key = path, 'artifact:' + path.name
        def download(self, target, overwrite):
            assert overwrite
            shutil.copytree(self.path, target, dirs_exist_ok=True)
    class Project:
        def log_artifact(self, name, kind, source):
            assert name == '4-statistics' and kind == 'artifact'
            assert (Path(source)/'analysis_status.json').is_file()
            self.manifest = json.loads((Path(source)/'digitalhub_inputs.json').read_text())
    monkeypatch.chdir(tmp_path)
    project = Project()
    module.analyze_statistics(project, **{name: Artifact(path) for name, path in paths.items()})
    assert set(project.manifest['inputs']) == set(paths)


def test_top64_single_slide_four_head_regression():
    # Execute the actual 2D normalization branch without importing torch/TRIDENT
    # (the statistics environment intentionally has neither dependency).
    path = ROOT.parent/'3-analysis/upstream/compute_abmil_layer_shift_repeated.py'
    tree = ast.parse(path.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'attn.ndim == 2')
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)
            self.shape = self.values.shape
        def mean(self, dim, keepdim=False):
            return Tensor(self.values.mean(axis=dim, keepdims=keepdim))
        def unsqueeze(self, dim):
            return Tensor(np.expand_dims(self.values, dim))
        def transpose(self, a, b):
            return Tensor(self.values.swapaxes(a, b))
    heads = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1],
                      [2, 3, 4, 5, 6], [0, 0, 0, 0, 0]])
    ns = dict(attn=Tensor(heads), bsz=1, n_tok=5)
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(path), 'exec'), ns)
    assert ns['attn'].shape == (1, 5)
    np.testing.assert_allclose(ns['attn'].values, [[2, 2.25, 2.5, 2.75, 3]])
