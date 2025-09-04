"""
Flux d'authentification 'site' : inscription, login web (cookie), accès aux pages protégées.
"""

import uuid


def test_register_and_web_login_and_access(client):
    # 1) Inscription
    suffix = uuid.uuid4().hex[:8]
    payload = {
        "username": f"user_{suffix}",
        "email": f"user_{suffix}@example.com",
        "password": "Password!",
    }
    r = client.post("/users", json=payload)
    assert r.status_code == 200
    assert r.json()["username"] == payload["username"]

    # 2) Login web -> pose cookie
    r2 = client.post(
        "/auth/web/token",
        data={"username": payload["username"], "password": payload["password"]},
    )
    assert r2.status_code == 200
    assert r2.json().get("access_token") is True
    assert "auth_token=" in r2.headers.get("set-cookie", "")

    # 3) Accès UI protégée
    r3 = client.get("/")
    assert r3.status_code == 200
    assert "text/html" in r3.headers.get("content-type", "")

    r4 = client.get("/data")
    assert r4.status_code == 200
    assert "text/html" in r4.headers.get("content-type", "")

    # 4) Logout
    r5 = client.post("/logout")
    assert r5.status_code == 200

    r6 = client.get("/", follow_redirects=False)
    assert r6.status_code in (302, 303, 307)
    assert r6.headers["location"] == "/login"
