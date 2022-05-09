import pytest
import sys

sys.path.insert(0, "src")

from app import app


@pytest.fixture
def client():
    app.config.update({"TESTING": True})

    with app.test_client() as client:
        yield client


def test_failure(client):
    response = client.get("/null")
    assert response.status_code == 404

def test_post_success(client):
    response = client.post(
        "/predict",
        query_string={"url": "https://dictionnaire.reverso.net/francais-arabe/"},
    )
    assert response.status_code == 200
    assert response.json["response"]["tags"] == "['108', '1265', '1494', '474', '692']"
