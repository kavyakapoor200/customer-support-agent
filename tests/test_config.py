"""Tests for configuration settings and thresholds validation."""
import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.core.thresholds import ActionThresholds, load_thresholds


def test_load_real_thresholds_file():
    """Validates that the production config/thresholds.yaml loads and passes validation."""
    config = load_thresholds("config/thresholds.yaml")
    assert config.version == "1.0"
    assert config.defaults.auto_execute_threshold == 0.90
    assert "refund" in config.policies

    # Refund policy check
    refund_policy = config.get_policy("refund")
    assert refund_policy.auto_execute_threshold == 0.90
    assert refund_policy.max_auto_amount_usd == 50.00
    assert refund_policy.review_threshold == 0.60
    assert refund_policy.deny_threshold == 0.60

    # P0 Security escalation check
    escalation_policy = config.get_policy("account_escalation")
    assert escalation_policy.p0_always_review is True

    # Fallback to defaults for unknown action
    unknown_policy = config.get_policy("non_existent_action")
    assert unknown_policy == config.defaults


def test_action_thresholds_hierarchy_valid():
    """Validates that valid descending thresholds pass."""
    threshold = ActionThresholds(
        auto_execute_threshold=0.90,
        max_auto_amount_usd=50.0,
        review_threshold=0.70,
        deny_threshold=0.50,
    )
    assert threshold.auto_execute_threshold == 0.90
    assert threshold.review_threshold == 0.70
    assert threshold.deny_threshold == 0.50


def test_action_thresholds_invalid_auto_less_than_review():
    """Asserts validation error when auto_execute_threshold is less than review_threshold."""
    with pytest.raises(ValidationError) as exc_info:
        ActionThresholds(
            auto_execute_threshold=0.60,
            review_threshold=0.80, # Invalid: review > auto
            deny_threshold=0.50,
        )
    assert "must be >= review_threshold" in str(exc_info.value)


def test_action_thresholds_invalid_review_less_than_deny():
    """Asserts validation error when review_threshold is less than deny_threshold."""
    with pytest.raises(ValidationError) as exc_info:
        ActionThresholds(
            auto_execute_threshold=0.90,
            review_threshold=0.40,
            deny_threshold=0.60, # Invalid: deny > review
        )
    assert "must be >= deny_threshold" in str(exc_info.value)


def test_action_thresholds_out_of_bounds():
    """Asserts validation error when thresholds are outside [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        ActionThresholds(
            auto_execute_threshold=1.50, # Invalid: > 1.0
            review_threshold=0.70,
            deny_threshold=0.50,
        )


def test_settings_env_override(monkeypatch):
    """Asserts environment variables properly override default settings."""
    monkeypatch.setenv("DECISION_ENGINE_BACKEND", "kev")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("GROQ_API_KEY", "test_key_123")

    settings = Settings()
    assert settings.DECISION_ENGINE_BACKEND == "kev"
    assert settings.PORT == 9000
    assert settings.GROQ_API_KEY == "test_key_123"
