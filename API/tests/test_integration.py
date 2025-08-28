import pytest

def test_home_redirects_to_login(client):
    r = client.get("/", allow_redirects=False)
    assert r.status_code in (302, 303, 307)
    assert r.headers["location"] == "/login"


def test_login_page_returns_html(client):
    r = client.get("/login")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


def test_logout_clears_cookie(client):
    r = client.post("/logout")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

