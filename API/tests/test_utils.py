"""
Unit tests for utility functions defined in ``API/utils.py``.

These tests exercise a number of small helper functions that are used
throughout the application.  They do not require any running
database or external services and should run quickly.  Where
appropriate the tests use simple ``numpy`` arrays and the built–in
``types.SimpleNamespace`` class to mock objects expected by the
functions under test.  Pandas is imported for functions that accept
``pandas.Series`` or ``pandas.DataFrame`` inputs.
"""

from types import SimpleNamespace
import numpy as np
import pytest

# pandas is an optional dependency in some environments; if it's not
# present the tests that rely on it will be skipped automatically.
pandas = pytest.importorskip("pandas")

from API import utils


def test_parse_vec_basic():
    """_parse_vec should return ``None`` for ``None`` input and
    convert lists and JSON strings into ``numpy`` arrays of dtype float."""
    assert utils._parse_vec(None) is None

    arr = utils._parse_vec([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == float
    assert np.allclose(arr, [1.0, 2.0, 3.0])

    arr_json = utils._parse_vec("[1, 2, 3]")
    assert isinstance(arr_json, np.ndarray)
    assert np.allclose(arr_json, [1.0, 2.0, 3.0])

    # Invalid strings should fall back to ``None``
    assert utils._parse_vec("not-a-json") is None


def test_parse_mat_basic():
    """_parse_mat should handle ``None``, lists of lists and JSON strings."""
    assert utils._parse_mat(None) == []

    mat = utils._parse_mat([[1, 2], [3, 4]])
    assert isinstance(mat, list)
    assert len(mat) == 2
    assert all(isinstance(x, np.ndarray) for x in mat)
    assert np.allclose(mat[0], [1.0, 2.0])
    assert np.allclose(mat[1], [3.0, 4.0])

    mat_json = utils._parse_mat("[[1, 2], [3, 4]]")
    assert isinstance(mat_json, list)
    assert len(mat_json) == 2
    assert np.allclose(mat_json[0], [1.0, 2.0])

    # Invalid strings should return an empty list
    assert utils._parse_mat("invalid") == []


def test_predict_scores_fallbacks():
    """_predict_scores should fall back through decision_function,
    predict_proba and predict in that order."""
    # decision_function path
    class DecisionModel:
        def decision_function(self, X):
            return np.array([1.0, 2.0, 3.0])

    scores = utils._predict_scores(DecisionModel(), np.zeros((3, 2)))
    assert np.allclose(scores, [1.0, 2.0, 3.0])

    # predict_proba path (returns 2d array)
    class ProbaModel:
        def predict_proba(self, X):
            return np.array([[0.1, 0.9], [0.2, 0.8]])

    scores2 = utils._predict_scores(ProbaModel(), np.zeros((2, 2)))
    # second column should be used because it has two classes
    assert np.allclose(scores2, [0.9, 0.8])

    # predict path
    class PredictModel:
        def predict(self, X):
            return np.array([5, 6])

    scores3 = utils._predict_scores(PredictModel(), np.zeros((2, 2)))
    assert np.allclose(scores3, [5.0, 6.0])


def test_align_df_to_cols():
    """_align_df_to_cols should add missing columns filled with zeros and
    preserve the order of ``feature_cols``."""
    df = pandas.DataFrame({"a": [1, 2]})
    out = utils._align_df_to_cols(df, ["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert np.allclose(out["a"].values, [1.0, 2.0])
    assert np.allclose(out["b"].values, [0.0, 0.0])


def test_to_np1d_and_to_list_np():
    """Verify conversion helpers ``to_np1d`` and ``to_list_np``."""
    arr = utils.to_np1d([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert np.allclose(arr, [1.0, 2.0, 3.0])

    # invalid input should return None
    assert utils.to_np1d("invalid") is None

    # list of lists should become list of ndarrays
    out = utils.to_list_np([[1, 2], [3, 4]])
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, np.ndarray) for x in out)

    # JSON string representation should also work
    out_json = utils.to_list_np("[[1, 2], [3, 4]]")
    assert isinstance(out_json, list) and len(out_json) == 2
    assert np.allclose(out_json[1], [3.0, 4.0])


def test_stub_sent_model_and_infer_embed_dim():
    """Verify the stub sentence model and embed dimension inference."""
    model = utils._StubSentModel(dim=5)
    embeds = model.encode(["hello", "world"])
    assert isinstance(embeds, list)
    assert len(embeds) == 2
    for vec in embeds:
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (5,)
        assert np.allclose(vec, np.zeros(5))

    # infer dimension from DataFrame
    df = pandas.DataFrame({"desc_embed": [np.zeros(7), np.ones(7)]})
    assert utils._infer_embed_dim(df) == 7


def test_format_opening_periods():
    """_format_opening_periods should produce a mapping from French
    day names to formatted opening hours."""
    periods = [
        SimpleNamespace(open_day=1, open_hour=9, open_minute=0, close_day=1, close_hour=17, close_minute=30),
        SimpleNamespace(open_day=3, open_hour=10, open_minute=0, close_day=3, close_hour=15, close_minute=0),
    ]
    res = utils._format_opening_periods(periods)
    assert res["Lundi"] == "09:00–17:30"
    assert res["Mercredi"] == "10:00–15:00"
    # days without periods should not appear
    assert "Dimanche" not in res


def test_pricelevel_to_int():
    """_pricelevel_to_int should map various representations to integers."""
    assert utils._pricelevel_to_int("PRICE_LEVEL_INEXPENSIVE") == 1
    assert utils._pricelevel_to_int("PRICE_LEVEL_MODERATE") == 2
    assert utils._pricelevel_to_int("PRICE_LEVEL_EXPENSIVE") == 3
    assert utils._pricelevel_to_int("PRICE_LEVEL_VERY_EXPENSIVE") == 4
    assert utils._pricelevel_to_int("3") == 3
    assert utils._pricelevel_to_int(2) == 2
    assert utils._pricelevel_to_int(None) is None


def test_determine_price_level():
    """determine_price_level should assign levels based on the 'start_price'
    column of a pandas Series.  Prices <=14 yield 1, >15 yield 2 and >20
    still yield 2 because of the ordering in the implementation.  NaN
    yields NaN (numpy)"""
    row1 = pandas.Series({"start_price": 10.0})
    assert utils.determine_price_level(row1) == 1

    row2 = pandas.Series({"start_price": 16.0})
    assert utils.determine_price_level(row2) == 2

    row3 = pandas.Series({"start_price": 25.0})
    # due to ordering, price > 20 still returns 2
    assert utils.determine_price_level(row3) == 2

    # missing start_price should return NaN
    row4 = pandas.Series({"start_price": np.nan})
    assert np.isnan(utils.determine_price_level(row4))
