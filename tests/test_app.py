import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_home_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200


def test_health_check(client):
    response = client.get("/health")
    data = response.get_json()

    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_json_prediction_returns_species(client):
    response = client.post(
        "/prediction",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    data = response.get_json()

    assert response.status_code == 200
    assert data["prediction"] == "setosa"


def test_json_prediction_rejects_missing_measurement(client):
    response = client.post(
        "/prediction",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
        },
    )

    assert response.status_code == 400
    assert "error" in response.get_json()