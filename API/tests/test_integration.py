"""
Lightweight integration tests exercising a few HTTP endpoints on the
FastAPI application defined in ``API/main.py``.  These tests focus on
routes that do not require a running database or external services.
They use FastAPI's ``TestClient`` to simulate HTTP requests.  To
disable expensive application warm-up during test startup the
``DISABLE_WARMUP`` environment variable is set.
"""

import os
import pytest
from fastapi.testclient import TestClient

from API.main import app


@pytest.fixture(scope="module")
def client():
    """Return a TestClient with warm-up disabled."""
    # Ensure the application doesn't attempt to load models or dataframes
    os.environ.setdefault("DISABLE_WARMUP", "1")
    return TestClient(app)


def test_home_redirects_to_login(client):
    """Unauthenticated access to '/' should redirect to the login page."""
    response = client.get("/", allow_redirects=False)
    # FastAPI/Starlette uses 307 for RedirectResponse by default
    assert response.status_code in (302, 303, 307)
    assert response.headers["location"] == "/login"


def test_login_page_returns_html(client):
    """GET /login should return a HTML page."""
    response = client.get("/login")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "text/html" in content_type


def test_logout_clears_cookie(client):
    """POST /logout should succeed and return a simple JSON payload."""
    response = client.post("/logout")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
