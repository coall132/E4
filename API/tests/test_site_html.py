"""
Tests 'site' (HTML) : vérifie les pages et formulaires côté UI.
"""

import os
import uuid
import pytest

bs4 = pytest.importorskip("bs4")
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


@pytest.fixture
def client(app_and_db):
    app, _, _ = app_and_db
    os.environ.setdefault("DISABLE_WARMUP", "1")
    return TestClient(app)


def test_login_page_has_form_and_fields(client):
    r = client.get("/login")
    assert r.status_code == 200
    soup = _soup(r.text)
    form = soup.find("form")
    assert form is not None
    # Le backend utilise OAuth2PasswordRequestForm -> 'username' & 'password'
    inputs = {i.get("name") for i in form.find_all("input")}
    assert "username" in inputs
    assert "password" in inputs


def test_register_page_has_form_and_fields(client):
    r = client.get("/register")
    assert r.status_code == 200
    soup = _soup(r.text)
    form = soup.find("form")
    assert form is not None
    inputs = {i.get("name") for i in form.find_all("input")}
    # On s'attend classiquement à username/email/password
    assert {"username", "email", "password"} <= inputs


def test_unauthenticated_pages_redirect(client):
    # '/' et '/predict' doivent protéger l'accès UI
    r1 = client.get("/", allow_redirects=False)
    assert r1.status_code in (302, 303, 307)
    assert r1.headers["location"] == "/login"

    r2 = client.get("/predict", allow_redirects=False)
    assert r2.status_code in (302, 303, 307)
    assert r2.headers["location"] == "/login"
