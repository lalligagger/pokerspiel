from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_summary_route_exposes_status_payload():
    response = client.get('/summary')
    assert response.status_code == 200, response.text
    data = response.json()
    assert 'solver' in data
    assert 'iteration' in data
    assert 'status' not in data or isinstance(data.get('status'), (dict, type(None)))
