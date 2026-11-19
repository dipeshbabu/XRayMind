import pytest
from fastapi.testclient import TestClient

from xraymind.api import app
from xraymind.cases import add_review, create_case, get_case_detail, update_case_status
from xraymind.store import SQLiteStore


@pytest.mark.parametrize('kwargs', [{'next_status': 'invalid'}, {'review_round': 0}, {'review_round': -1}])
def test_invalid_review_does_not_write_partial_data(tmp_path, kwargs):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    case = create_case('image.png', store=store)
    before = get_case_detail(case['id'], store=store)
    with pytest.raises(ValueError):
        add_review(case['id'], decision='agree', store=store, **kwargs)
    assert get_case_detail(case['id'], store=store) == before


def test_missing_case_operations_raise_key_error(tmp_path):
    store = SQLiteStore(tmp_path / 'cases.sqlite3')
    with pytest.raises(KeyError):
        update_case_status(123, 'reviewed', store=store)
    with pytest.raises(KeyError):
        add_review(123, decision='agree', store=store)
    assert store.fetch_all('SELECT * FROM audit_events') == []


@pytest.mark.parametrize('route,data', [('status', {'status': 'reviewed'}), ('review', {'decision': 'agree'})])
def test_missing_case_api_returns_404(tmp_path, monkeypatch, route, data):
    monkeypatch.setenv('XRAYMIND_DB_PATH', str(tmp_path / 'cases.sqlite3'))
    monkeypatch.delenv('XRAYMIND_API_KEYS', raising=False)
    monkeypatch.delenv('XRAYMIND_API_KEY', raising=False)
    with TestClient(app) as client:
        assert client.post(f'/cases/123/{route}', data=data).status_code == 404
