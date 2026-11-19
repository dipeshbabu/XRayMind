import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from xraymind import cases, hosted_jobs, packet, tta
from xraymind.api_auth import _parse_role_keys
from xraymind.conformal import apply_conformal_sets, calibrate_conformal_thresholds, conformal_quantile, split_calibration_eval
from xraymind.dashboard import cases_requiring_attention
from xraymind.evaluation import evaluate_predictions, evaluate_subgroups
from xraymind.report import save_html_report
from xraymind.selective import evaluate_selective_predictions, selective_curve_for_label
from xraymind.store import SQLiteStore


def test_tta_aligns_scores_by_label(monkeypatch):
    monkeypatch.setattr(tta, '_variants', lambda image: [0, 1])
    results = [
        [{'label': 'A', 'probability': 0.9}, {'label': 'B', 'probability': 0.1}],
        [{'label': 'B', 'probability': 0.8}, {'label': 'A', 'probability': 0.2}],
    ]
    monkeypatch.setattr(tta, 'predict_image', lambda image, **kwargs: {'predictions': results[image]})
    result = tta.predict_with_tta(None, 'test')
    values = {row['label']: row for row in result['predictions']}
    assert values['A']['probability_mean'] == pytest.approx(0.55)
    assert values['B']['probability_mean'] == pytest.approx(0.45)
    assert values['A']['probability_std'] == pytest.approx(0.35)


def test_evaluation_never_uses_truth_as_predictions():
    labels = pd.DataFrame({'image': ['a', 'b'], 'A': [0, 1], 'missing': [0, 1], 'group': ['x', 'x']})
    predictions = pd.DataFrame({'image': ['a', 'b'], 'A': [0.2, 0.8]})
    assert list(evaluate_predictions(labels, predictions)['label']) == ['A']
    assert set(evaluate_selective_predictions(labels, predictions)['label']) == {'A'}
    assert list(calibrate_conformal_thresholds(labels, predictions, labels=['A', 'missing'])['label']) == ['A']
    assert list(evaluate_subgroups(labels, predictions, ['group'], min_group_size=2)['label']) == ['A']


def test_conformal_small_sample_includes_both_labels():
    qhat = conformal_quantile([0.1], alpha=0.1)
    assert np.isinf(qhat)
    result = apply_conformal_sets(pd.DataFrame({'A': [0.0, 0.5, 1.0]}), pd.DataFrame({'label': ['A'], 'qhat': [qhat]}))
    assert result['A_conformal_set_size'].tolist() == [2, 2, 2]


@pytest.mark.parametrize('size', [0, 1])
def test_conformal_split_rejects_insufficient_data(size):
    with pytest.raises(ValueError):
        split_calibration_eval(pd.DataFrame({'image': list(range(size))}))


def test_selective_accepts_numpy_coverage_grid():
    result = selective_curve_for_label([0, 1], [0.1, 0.9], 'A', coverage_grid=np.array([0.5, 1.0]))
    assert result['coverage'].tolist() == [0.5, 1.0]


@pytest.mark.parametrize('coverage', [0, -0.1, 1.1, float('nan')])
def test_selective_rejects_invalid_coverage(coverage):
    with pytest.raises(ValueError):
        selective_curve_for_label([0, 1], [0.1, 0.9], 'A', coverage_grid=[coverage])


def test_report_media_links_resolve_from_report_directory(tmp_path):
    image = tmp_path / 'images' / 'preview.png'
    image.parent.mkdir()
    Image.new('L', (4, 4)).save(image)
    report = save_html_report({}, tmp_path / 'reports' / 'report.html', original_path=image)
    assert "src='../images/preview.png'" in report.read_text()


def test_packet_archive_has_final_manifest_and_portable_images(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, 'predict_image', lambda **kwargs: {'top_findings': []})
    result = packet.create_study_packet(np.zeros((8, 8), dtype=np.uint8), tmp_path / 'packet', 'test')
    with ZipFile(result['files']['zip']) as archive:
        assert json.loads(archive.read('manifest.json')) == result
        assert "src='original_preview.png'" in archive.read('report.html').decode()


def test_failed_inference_does_not_create_case(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    def fail(*args, **kwargs):
        raise ValueError('bad image')
    monkeypatch.setattr(cases, 'predict_image', fail)
    with pytest.raises(ValueError):
        cases.create_case_with_prediction('bad.png', store=store)
    assert cases.list_cases(store=store) == []


def test_reviewer_filter_applies_before_limit(tmp_path):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    target = cases.create_case('target.png', assigned_to='reader', store=store)
    for index in range(5):
        cases.create_case(f'{index}.png', assigned_to='other', priority='urgent', store=store)
    queue = cases_requiring_attention(store=store, reviewer='reader', include_unassigned=False, limit=1)
    assert [item['id'] for item in queue] == [target['id']]
    cases.assign_case(target['id'], reviewer='reader', needs_second_reader=True, store=store)
    cases.update_case_status(target['id'], 'archived', store=store)
    assert cases_requiring_attention(store=store, reviewer='reader', include_unassigned=False) == []


def test_workers_claim_job_once_and_running_job_cannot_be_cancelled(tmp_path, monkeypatch):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    job = hosted_jobs.enqueue_case_prediction_job('image.png', store=store)
    started = threading.Event()
    finish = threading.Event()
    calls = []
    def run(job, **kwargs):
        calls.append(job['id'])
        started.set()
        assert finish.wait(10)
        return {}
    monkeypatch.setattr(hosted_jobs, '_run_job', run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(hosted_jobs.process_next_job, store=store)
        try:
            assert started.wait(10)
            assert pool.submit(hosted_jobs.process_next_job, store=store).result(timeout=10) is None
            with pytest.raises(ValueError):
                hosted_jobs.cancel_job(job['id'], store=store)
        finally:
            finish.set()
        assert first.result(timeout=10)['status'] == 'succeeded'
    assert calls == [job['id']]
    assert hosted_jobs.get_job(job['id'], store=store)['attempts'] == 1


@pytest.mark.parametrize('config', ['reviewer:abc==', 'reader=reviewer:abc=='])
def test_api_keys_allow_base64_padding(config):
    assert _parse_role_keys(config)['abc=='].role == 'reviewer'


@pytest.mark.parametrize('config', [' ', ',,,'])
def test_empty_role_key_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        _parse_role_keys(config)


def test_threshold_tuning_aligns_images(tmp_path, monkeypatch):
    from argparse import Namespace
    from importlib.util import module_from_spec, spec_from_file_location

    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'tune_thresholds.py'
    spec = spec_from_file_location('tune_thresholds', script_path)
    tune_thresholds = module_from_spec(spec)
    spec.loader.exec_module(tune_thresholds)

    labels = tmp_path / 'labels.csv'
    predictions = tmp_path / 'predictions.csv'
    output = tmp_path / 'thresholds.csv'
    pd.DataFrame({'image': ['a', 'b'], 'A': [0, 1]}).to_csv(labels, index=False)
    pd.DataFrame({'image': ['b', 'a'], 'A': [0.8, 0.2]}).to_csv(predictions, index=False)
    monkeypatch.setattr(tune_thresholds, 'parse_args', lambda: Namespace(
        labels=labels, predictions=predictions, out=output, image_column='image', objective='f1',
    ))
    def tune(y_true, y_score, **kwargs):
        assert list(y_true) == [0, 1]
        assert list(y_score) == [0.2, 0.8]
        return 0.5
    monkeypatch.setattr(tune_thresholds, 'tune_threshold', tune)
    assert tune_thresholds.main() == 0
    assert pd.read_csv(output).loc[0, 'threshold'] == 0.5


def test_model_loader_dispatches_model_architecture(monkeypatch):
    from xraymind import model_loader
    calls = []
    class Model:
        def eval(self):
            calls.append('eval')
    def get_model(name):
        calls.append(name)
        return Model()
    monkeypatch.setattr(model_loader.xrv.models, 'get_model', get_model)
    model_loader.load_model.cache_clear()
    try:
        model_loader.load_model('resnet50-res512-all')
        assert calls == ['resnet50-res512-all', 'eval']
    finally:
        model_loader.load_model.cache_clear()


def test_uploads_with_same_name_remain_distinct(tmp_path, monkeypatch):
    from io import BytesIO
    from fastapi import UploadFile
    from xraymind.api import _persist_upload

    monkeypatch.setenv('XRAYMIND_CASE_UPLOAD_DIR', str(tmp_path))
    def upload(value):
        return _persist_upload(UploadFile(filename='image.png', file=BytesIO(value)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(upload, [b'one', b'two']))
    assert paths[0] != paths[1]
    assert [path.read_bytes() for path in paths] == [b'one', b'two']


@pytest.mark.parametrize('kwargs', [{'top_k': 0}, {'top_k': -1}, {'threshold': -0.1}, {'threshold': float('nan')}])
def test_invalid_prediction_options_fail_before_model_load(monkeypatch, kwargs):
    from xraymind import inference
    monkeypatch.setattr(inference, 'load_model', lambda name: pytest.fail('should not load weights'))
    with pytest.raises(ValueError):
        inference.predict_image('unused.png', **kwargs)
