"""
Flux d'authentification 'site' : inscription, login web (cookie), accès aux pages protégées.
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(app_and_db):
    app, _, _ = app_and_db
    os.environ.setdefault("DISABLE_WARMUP", "1")
    return TestClient(app)


def test_register_and_web_login_and_access(client):
    # 1) Inscription utilisateur
    suffix = uuid.uuid4().hex[:8]
    payload = {
        "username": f"user_{suffix}",
        "email": f"user_{suffix}@example.com",
        "password": "P@ssw0rd!",
    }
    r = client.post("/users", json=payload)
    assert r.status_code == 200
    user = r.json()
    assert user["username"] == payload["username"]

    # 2) Login web -> pose un cookie 'auth_token'
    #    (OAuth2PasswordRequestForm attend un form-encoded)
    r2 = client.post(
        "/auth/web/login",
        data={"username": payload["username"], "password": payload["password"]},
    )
    assert r2.status_code == 200
    assert r2.json().get("ok") is True
    set_cookie = r2.headers.get("set-cookie", "")
    assert "auth_token=" in set_cookie

    # 3) Accès à '/', 'home', 'data' avec le cookie
    r3 = client.get("/")
    assert r3.status_code == 200
    assert "text/html" in r3.headers.get("content-type", "")

    r4 = client.get("/data")
    assert r4.status_code == 200
    assert "text/html" in r4.headers.get("content-type", "")

    # 4) Logout -> efface le cookie
    r5 = client.post("/logout")
    assert r5.status_code == 200
    # Après logout, retour au login
    r6 = client.get("/", allow_redirects=False)
    assert r6.status_code in (302, 303, 307)
    assert r6.headers["location"] == "/login"
