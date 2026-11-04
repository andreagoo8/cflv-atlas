import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(folder):
    spec = importlib.util.spec_from_file_location(folder.replace('-', '_'), ROOT/folder/'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Artifact:
    def __init__(self, path):
        self.path, self.key = path, 'artifact:'+path.name
    def download(self, dest, overwrite):
        shutil.copytree(self.path, dest, dirs_exist_ok=True)


class Project:
    def __init__(self):
        self.logged = []
    def log_artifact(self, **kwargs):
        self.logged.append(kwargs)


def cohorts(tmp, subdir):
    root=tmp/'features'
    (root/subdir).mkdir(parents=True)
    (root/'labels.csv').write_text('slide_name,slide_label\na,0\nb,1\n')
    for name in ['a','b']:
        (root/subdir/f'{name}.h5').touch()
    return {key:Artifact(root) for key in ['train_features','val_features','internal_test_features','external_test_features']}


def options(cmd):
    return {cmd[i][2:]:cmd[i+1] for i in range(3,len(cmd)-1) if cmd[i].startswith('--') and not cmd[i+1].startswith('--')}


@pytest.mark.parametrize('mode,pooling,subdir',[
    ('patch','abmil','features_conch_v15'),('patch','mean','features_conch_v15'),
    ('patch','max','features_conch_v15'),('slide','abmil','slide_features_prism'),
    ('slide','abmil','slide_features_titan')])
def test_training_dispatch(tmp_path,monkeypatch,mode,pooling,subdir):
    module=load('2-abmil');project=Project();monkeypatch.chdir(tmp_path)
    def execute(cmd,check):
        opts=options(cmd)
        assert opts['mode']==mode and opts['tile_pooling']==pooling
        assert opts['feature_subdir']==subdir
        assert '--use_scheduler' in cmd
        out=Path(opts['output_root'])/opts['run_name']
        for name in ['best_model.pt','config.json','history.json','metrics.json','summary.csv',
                     'preds_val.csv','preds_internal_test.csv','preds_external_test.csv']:
            (out/name).touch()
    monkeypatch.setattr(module.subprocess,'run',execute)
    module.train_abmil(project,**cohorts(tmp_path,subdir),mode=mode,tile_pooling=pooling,feature_subdir=subdir)
    assert len(project.logged)==1


def test_single_class_training_is_rejected_before_execution(tmp_path, monkeypatch):
    module = load('2-abmil')
    project = Project()
    monkeypatch.chdir(tmp_path)
    inputs = cohorts(tmp_path, 'features')
    (tmp_path/'features'/'labels.csv').write_text('slide_name,slide_label\na,0\nb,0\n')
    def execute(*args, **kwargs):
        pytest.fail('Single-class inputs must be rejected before starting training')
    monkeypatch.setattr(module.subprocess, 'run', execute)
    with pytest.raises(ValueError, match='Training requires at least two classes'):
        module.train_abmil(project, **inputs, feature_subdir='features')
    assert not project.logged


@pytest.mark.parametrize('encoder,patch,size,cache', [('prism','virchow',224,False),('titan','conch_v15',512,True)])
def test_slide_extraction_dispatch(tmp_path,monkeypatch,encoder,patch,size,cache):
    module=load('1-trident');project=Project();monkeypatch.chdir(tmp_path)
    slides=tmp_path/'slides';slides.mkdir();(slides/'a.svs').touch()
    labels=tmp_path/'labels';labels.mkdir();(labels/'labels.csv').write_text('slide_name,slide_label\na,0\n')
    cached=[]
    monkeypatch.setattr(module,'_prepare_conch_v15_cache',lambda env:cached.append(True))
    def execute(cmd,check,env):
        assert cmd[cmd.index('--slide_encoder')+1]==encoder
        assert cmd[cmd.index('--patch_size')+1]==str(size)
        assert cmd[cmd.index('--patch_encoder')+1]==patch
    monkeypatch.setattr(module.subprocess,'run',execute)
    module.trident_feature_extraction(project,Artifact(slides),Artifact(labels),'features',
                                      slide_encoder=encoder,patch_encoder=patch,patch_size=size)
    assert bool(cached)==cache
    assert len(project.logged)==1


@pytest.mark.parametrize('mode,pooling', [('patch','abmil'),('patch','mean'),('patch','max'),('slide','abmil')])
def test_checkpoint_config_overrides_upstream_defaults(tmp_path,monkeypatch,mode,pooling):
    module=load('2-inference');project=Project();monkeypatch.chdir(tmp_path)
    training=tmp_path/'saved';training.mkdir()
    cfg=dict(run_name='saved',mode=mode,tile_pooling=pooling,feature_subdir='features',seed=42,
             num_patches=2000,hidden_dim=512,dropout=.2,n_heads=4,head_dim=256,no_gated=True)
    (training/'config.json').write_text(json.dumps(cfg));(training/'best_model.pt').touch()
    def execute(cmd,check):
        opts=options(cmd)
        for key,value in cfg.items():
            if key not in ['run_name','no_gated']:assert opts[key]==str(value)
        assert '--no_gated' in cmd and '--save_train' in cmd
        assert Path(opts['checkpoint']).is_file()
        for split in ['train','val','internal_test','external_test']:
            assert (Path(opts[split+'_root'])/'labels.csv').is_file()
        out=Path(opts['output_dir']);out.mkdir()
        for name in ['metrics.json','summary.csv','preds_val.csv','preds_internal_test.csv',
                     'preds_external_test.csv','preds_train.csv']:(out/name).touch()
    monkeypatch.setattr(module.subprocess,'run',execute)
    module.predict_classifier(project,Artifact(training),**cohorts(tmp_path,'features'),save_train=True)
    logged=Path(project.logged[0]['source'])
    assert (logged/'best_model.pt').is_file()
    assert len(list(logged.glob('predictions_*/digitalhub_inputs.json')))==1


def test_ambiguous_checkpoint_run_is_rejected(tmp_path,monkeypatch):
    module=load('2-inference');monkeypatch.chdir(tmp_path);project=Project()
    for name in ['one','two']:
        path=tmp_path/'saved'/name;path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps(dict(run_name=name)))
    with pytest.raises(ValueError,match='exactly one'):
        module.predict_classifier(project,Artifact(tmp_path/'saved'),**cohorts(tmp_path,'features'))
    assert not project.logged
