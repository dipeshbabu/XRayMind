import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from xraymind import api, cases, cli


@pytest.fixture
def image_bytes():
    buffer = io.BytesIO()
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('XRAYMIND_DB_PATH', str(tmp_path / 'cases.sqlite3'))
    monkeypatch.setenv('XRAYMIND_CASE_UPLOAD_DIR', str(tmp_path / 'uploads'))
    monkeypatch.setenv('XRAYMIND_API_KEYS', 'ops=admin:admin,reader=reviewer:reader,other=reviewer:other,audit=viewer:viewer')
    monkeypatch.delenv('XRAYMIND_API_KEY', raising=False)
    def prediction(image, **kwargs):
        with Image.open(image) as opened:
            opened.verify()
        return {'model': 'test', 'predictions': [{'label': 'A', 'probability': 0.8, 'positive': True}],
                'top_findings': [{'label': 'A', 'probability': 0.8, 'positive': True}],
                'uncertainty': {'max_probability': 0.8, 'low_confidence': False}}
    monkeypatch.setattr(cases, 'predict_image', prediction)
    monkeypatch.setattr(api, 'predict_image', prediction)
    with TestClient(api.app) as test_client:
        yield test_client


def headers(key):
    return {'X-API-Key': key}


def test_api_case_review_export_monitor_roundtrip(client, tmp_path, image_bytes):
    assert client.get('/me').status_code == 401
    assert client.get('/me', headers=headers('reader')).json() == {'key_id': 'reader', 'role': 'reviewer'}
    files = {'file': ('image.png', image_bytes, 'image/png')}
    assert client.post('/cases', headers=headers('viewer'), files=files).status_code == 403
    created = client.post('/cases', headers=headers('reader'), files=files)
    assert created.status_code == 200
    case_id = created.json()['case']['id']
    assert Path(created.json()['case']['image_path']).read_bytes() == image_bytes
    assert client.post('/reviewer/claim', headers=headers('reader'), data={'case_id': case_id}).status_code == 200
    queue = client.get('/reviewer/queue?include_unassigned=false', headers=headers('reader')).json()
    assert [case['id'] for case in queue['cases']] == [case_id]
    assert client.post('/reviewer/release', headers=headers('other'), data={'case_id': case_id}).status_code == 403
    reviewed = client.post(f'/cases/{case_id}/review', headers=headers('reader'), data={'decision': 'uncertain'})
    assert reviewed.status_code == 200
    assert reviewed.json()['case_detail']['case']['needs_second_reader'] is True
    assert client.post('/reviewer/release', headers=headers('reader'), data={'case_id': case_id}).json()['case']['assigned_to'] is None
    assert client.get('/cases', headers=headers('viewer')).json()['count'] == 1
    assert client.get('/dashboard/summary', headers=headers('viewer')).json()['reviewed_cases'] == 1
    export_data = {'out_dir': str(tmp_path / 'exports')}
    assert client.post('/exports/cases', headers=headers('reader'), data=export_data).status_code == 403
    export = client.post('/exports/cases', headers=headers('admin'), data=export_data).json()
    assert export['row_count'] == 1
    for kind in ['csv', 'jsonl']:
        assert hashlib.sha256(Path(export['files'][kind]).read_bytes()).hexdigest() == export['sha256'][kind]
    snapshot = client.get('/monitoring/snapshot', headers=headers('viewer')).json()
    assert snapshot['total_cases'] == 1
    assert snapshot['reviewed_cases'] == 1
    report = client.get('/monitoring/report.md', headers=headers('viewer'))
    assert report.status_code == 200 and 'Validation Monitoring Report' in report.text


def test_api_job_tenant_and_lifecycle(client, image_bytes):
    created = client.post('/jobs/case-prediction', headers=headers('reader'),
                          files={'file': ('image.png', image_bytes, 'image/png')})
    assert created.status_code == 200
    job_id = created.json()['job']['id']
    assert client.get('/jobs', headers=headers('other')).json()['count'] == 0
    assert client.get(f'/jobs/{job_id}', headers=headers('other')).status_code == 403
    assert client.post('/jobs/process-next', headers=headers('reader')).status_code == 403
    processed = client.post('/jobs/process-next', headers=headers('admin')).json()['job']
    assert processed['status'] == 'succeeded'
    assert processed['result']['case_id'] == 1
    assert client.post(f'/jobs/{job_id}/cancel', headers=headers('reader')).status_code == 400
    assert client.post('/jobs/process-next', headers=headers('admin')).json()['job'] is None


def test_api_batch_preserves_success_when_one_image_is_invalid(client, tmp_path, image_bytes):
    response = client.post('/batch-predict', headers=headers('reader'),
                           data={'audit_log': str(tmp_path / 'audit.jsonl')},
                           files=[('files', ('good.png', image_bytes, 'image/png')),
                                  ('files', ('bad.png', b'bad image', 'image/png'))])
    assert response.status_code == 200
    assert [item['status'] for item in response.json()['results']] == ['success', 'error']
    assert len((tmp_path / 'audit.jsonl').read_text().splitlines()) == 2


def test_cli_persistence_and_export_roundtrip(tmp_path, capsys):
    db = tmp_path / 'cases.sqlite3'
    case = cases.create_case('sample.png', db_path=db)
    assert cli.main(['case', 'assign', str(case['id']), '--reviewer', 'reader', '--db', str(db)]) == 0
    assert json.loads(capsys.readouterr().out)['case']['assigned_to'] == 'reader'
    assert cli.main(['case', 'review', str(case['id']), '--decision', 'agree', '--db', str(db)]) == 0
    assert json.loads(capsys.readouterr().out)['case_detail']['case']['status'] == 'reviewed'
    assert cli.main(['export', '--db', str(db), '--out-dir', str(tmp_path / 'exports')]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest['row_count'] == 1
    assert cli.main(['monitor', '--db', str(db), '--no-write']) == 0
    assert json.loads(capsys.readouterr().out)['reviewed_cases'] == 1
