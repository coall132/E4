"""
Tests 'site' (HTML) : vérifie les pages et formulaires côté UI.
"""

import pytest
from bs4 import BeautifulSoup

bs4 = pytest.importorskip("bs4")


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_login_page_has_form_and_fields(client):
    r = client.get("/login")
    assert r.status_code == 200
    soup = _soup(r.text)
    form = soup.find("form")
    assert form is not None
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
    assert {"username", "email", "password"} <= inputs


def test_unauthenticated_pages_redirect(client):
    r1 = client.get("/", allow_redirects=False)
    assert r1.status_code in (302, 303, 307)
    assert r1.headers["location"] == "/login"

    r2 = client.get("/predict", allow_redirects=False)
    assert r2.status_code in (302, 303, 307)
    assert r2.headers["location"] == "/login"
