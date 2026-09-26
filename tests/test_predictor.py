import pytest

from app.predictor import RoutingPredictor, ValidationError


@pytest.fixture(scope="module")
def predictor():
    return RoutingPredictor.instance()


def test_model_loads(predictor):
    assert predictor.model is not None
    assert predictor.feature_bundle is not None


def test_predict_returns_exactly_one_team(predictor):
    result = predictor.predict({
        "request_text": "water purifier leaking from the bottom",
        "product_family": "Water Purifier", "warranty_status": "in_warranty", "channel": "chat",
    })
    assert isinstance(result["predicted_team"], str)
    assert result["predicted_team"] != ""


def test_predict_has_confidence_fields(predictor):
    result = predictor.predict({"request_text": "fan making noise"})
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence_band"] in ("high", "medium", "low")


def test_predict_has_review_flag(predictor):
    result = predictor.predict({"request_text": "please call me"})
    assert isinstance(result["review_required"], bool)


def test_predict_generates_nonempty_explanation(predictor):
    result = predictor.predict({"request_text": "purifier leaking water"})
    assert len(result["reasons"]) > 0
    assert all(isinstance(r, str) and r for r in result["reasons"])


def test_predict_handles_missing_optional_fields(predictor):
    # Should not crash if product_family/warranty_status/channel are absent.
    result = predictor.predict({"request_text": "need help with my appliance"})
    assert "predicted_team" in result


def test_empty_text_raises_validation_error(predictor):
    with pytest.raises(ValidationError):
        predictor.predict({"request_text": ""})


def test_missing_text_raises_validation_error(predictor):
    with pytest.raises(ValidationError):
        predictor.predict({})


def test_retrieval_fallback_returns_similar_cases(predictor):
    result = predictor.predict({"request_text": "purifier leaking from bottom"})
    assert isinstance(result["similar_cases"], list)
    if predictor.retriever is not None:
        assert len(result["similar_cases"]) > 0
        for case in result["similar_cases"]:
            assert "team" in case and "similarity" in case


def test_payment_mention_policy_case_matches_assignment_example(predictor):
    """The exact example from the assignment brief: payment method mentioned,
    but the real issue is a product fault, so it should NOT route to Billing."""
    result = predictor.predict({
        "request_text": "My purifier was paid for by UPI and now it is leaking",
        "product_family": "Water Purifier",
    })
    assert result["predicted_team"] == "Repairs"
