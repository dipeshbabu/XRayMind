import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from xraymind.api import app
from xraymind.cases import add_review, create_case, get_case_detail
from xraymind.conformal import binary_nonconformity_scores
from xraymind.store import SQLiteStore


def test_automatic_review_round_follows_highest_round(tmp_path):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    case = create_case('image.png', store=store)
    add_review(case['id'], decision='agree', review_round=5, store=store)
    review = add_review(case['id'], decision='agree', store=store)
    assert review['review_round'] == 6


def test_failed_review_update_rolls_back_review_and_audit(tmp_path):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    case = create_case('image.png', store=store)
    before = get_case_detail(case['id'], store=store)
    events = store.fetch_all('SELECT * FROM audit_events')
    store.execute("CREATE TRIGGER reject_status BEFORE UPDATE ON cases BEGIN SELECT RAISE(ABORT, 'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        add_review(case['id'], decision='agree', store=store)
    assert get_case_detail(case['id'], store=store) == before
    assert store.fetch_all('SELECT * FROM audit_events') == events


@pytest.mark.parametrize('route', ['/cases?limit=-1', '/cases?offset=-1', '/jobs?limit=-1', '/dashboard/attention?limit=-1'])
def test_api_rejects_negative_pagination(tmp_path, monkeypatch, route):
    monkeypatch.setenv('XRAYMIND_DB_PATH', str(tmp_path / 'cases.sqlite3'))
    monkeypatch.delenv('XRAYMIND_API_KEY', raising=False)
    monkeypatch.delenv('XRAYMIND_API_KEYS', raising=False)
    with TestClient(app) as client:
        assert client.get(route).status_code == 422


@pytest.mark.parametrize('options', [{'top_k': 0}, {'threshold': 1.1}, {'threshold': -0.1}])
def test_api_rejects_bad_prediction_options(monkeypatch, options):
    monkeypatch.delenv('XRAYMIND_API_KEY', raising=False)
    monkeypatch.delenv('XRAYMIND_API_KEYS', raising=False)
    with TestClient(app) as client:
        response = client.post('/predict', data=options, files={'file': ('image.png', b'not-an-image', 'image/png')})
    assert response.status_code == 422


@pytest.mark.parametrize('labels,scores', [([0.5], [0.2]), ([1], [np.nan]), ([1], [1.2]), ([0], [-0.1])])
def test_conformal_rejects_invalid_binary_data(labels, scores):
    with pytest.raises(ValueError):
        binary_nonconformity_scores(labels, scores)


def test_concurrent_reviews_receive_distinct_rounds(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    case = create_case('image.png', store=store)
    ready = Barrier(2)
    def review(reviewer):
        ready.wait(timeout=10)
        return add_review(case['id'], decision='agree', reviewer=reviewer, store=store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reviews = list(pool.map(review, ['one', 'two']))
    assert sorted(item['review_round'] for item in reviews) == [1, 2]
    detail = get_case_detail(case['id'], store=store)
    assert len(detail['reviews']) == 2
    assert detail['case']['status'] == 'reviewed'
