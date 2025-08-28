"""
Unit tests for selected functions in ``API/CRUD.py``.

These tests cover cryptographic helpers (API key generation and hashing),
JWT token issuance, subject parsing and several small formatting
functions.  The aim is to ensure deterministic behaviour for these
helpers without requiring a live database.  When external libraries
such as ``python-jose`` are not available the corresponding test will
be skipped automatically.
"""

from types import SimpleNamespace
import numpy as np
import pytest

from API import CRUD


def test_generate_api_key_format():
    """Ensure generate_api_key returns a properly formatted key and ids."""
    key, key_id, secret = CRUD.generate_api_key()
    # key should start with rk_<id>.secret
    assert key.startswith("rk_")
    assert "." in key
    prefix, dot, sec = key.partition(".")
    # prefix after rk_ should match key_id
    assert prefix.replace("rk_", "", 1) == key_id
    # secret returned separately should match the suffix
    assert sec == secret


def test_hash_and_verify_api_key():
    """Verify that a hashed API key can be verified and mismatched keys fail."""
    key, key_id, secret = CRUD.generate_api_key()
    hashed = CRUD.hash_api_key(key)
    assert CRUD.verify_api_key_hash(key, hashed) is True
    # altering the key should lead to verification failure
    assert CRUD.verify_api_key_hash(key + "x", hashed) is False


def test_create_access_token_and_decode():
    """Create a JWT and decode it to verify the subject and exp claims."""
    jwt = pytest.importorskip("jose.jwt")
    subject = "user:42"
    token, exp_ts = CRUD.create_access_token(subject)
    assert isinstance(token, str)
    assert isinstance(exp_ts, int)
    decoded = jwt.decode(token, CRUD.JWT_SECRET, algorithms=[CRUD.ALGORITHM])
    assert decoded["sub"] == subject
    # exp claim should be present and reasonable
    assert isinstance(decoded["exp"], int)
    assert decoded["exp"] >= exp_ts - 1  # small leeway


def test_current_user_id_parsing():
    """Test that current_user_id extracts the integer id from the subject."""
    assert CRUD.current_user_id("user:99") == 99
    with pytest.raises(Exception):
        CRUD.current_user_id("invalid")


def test_price_to_int_and_symbol():
    """Ensure _price_to_int_and_symbol maps price representations to expected tuples."""
    assert CRUD._price_to_int_and_symbol("PRICE_LEVEL_INEXPENSIVE") == (1, "€")
    assert CRUD._price_to_int_and_symbol("PRICE_LEVEL_MODERATE") == (2, "€€")
    assert CRUD._price_to_int_and_symbol("PRICE_LEVEL_EXPENSIVE") == (3, "€€€")
    assert CRUD._price_to_int_and_symbol("PRICE_LEVEL_VERY_EXPENSIVE") == (4, "€€€€")
    assert CRUD._price_to_int_and_symbol("3") == (3, "€€€")
    assert CRUD._price_to_int_and_symbol(2) == (2, "€€")
    # Dollar strings should map by length
    assert CRUD._price_to_int_and_symbol("$") == (1, "€")
    assert CRUD._price_to_int_and_symbol("$$$$") == (4, "€€€€")
    # Unknown values return (None, None)
    assert CRUD._price_to_int_and_symbol("unknown") == (None, None)
    assert CRUD._price_to_int_and_symbol(None) == (None, None)


def test_fmt_time():
    """Test formatting of hour/minute combinations."""
    assert CRUD._fmt_time(10, 0) == "10h"
    assert CRUD._fmt_time(9, 30) == "9h30"
    assert CRUD._fmt_time(None, 15) is None


def test_build_horaires_lines():
    """_build_horaires should produce one line per day with either a list of slots or 'Fermé'."""
    # create sample periods for Monday (1) and Tuesday (2)
    periods = [
        SimpleNamespace(open_day=1, close_day=1, open_hour=9, open_minute=0, close_hour=12, close_minute=0),
        SimpleNamespace(open_day=1, close_day=1, open_hour=14, open_minute=0, close_hour=18, close_minute=30),
        SimpleNamespace(open_day=2, close_day=2, open_hour=10, open_minute=0, close_hour=11, close_minute=0),
    ]
    lines = CRUD._build_horaires(periods)
    # transform list of lines into a mapping {day: slots}
    mapping = dict(line.split("\t", 1) for line in lines)
    assert mapping["Lundi"] == "9h–12h, 14h–18h30"
    assert mapping["Mardi"] == "10h–11h"
    # other days should be marked as Fermé
    for day in ["Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]:
        assert mapping[day] == "Fermé" 


def test_numeric_bool_to_float():
    """_numeric_bool_to_float should retain numeric/bool columns and cast booleans to floats."""
    pandas = pytest.importorskip("pandas")
    df = pandas.DataFrame({"a": [1, 2], "b": [True, False], "c": ["x", "y"]})
    out = CRUD._numeric_bool_to_float(df)
    assert list(out.columns) == ["a", "b"]
    assert out["b"].dtype == float
    assert np.allclose(out["b"].values, [1.0, 0.0])
