from evolution.core.config import (
    get_missing_credentials_message,
    get_model_provider,
    load_hermes_provider_config,
    normalize_model_for_litellm,
)


def test_get_model_provider_parses_litellm_prefix():
    assert get_model_provider("openai/gpt-4.1") == "openai"
    assert get_model_provider("anthropic/claude-sonnet-4") == "anthropic"


def test_get_model_provider_returns_none_without_prefix():
    assert get_model_provider("gpt-4.1") is None
    assert get_model_provider("Qwen/Qwen3.6-35B-A3B-FP8") is None


def test_normalize_model_for_openai_compatible_endpoint():
    assert normalize_model_for_litellm(
        "Qwen/Qwen3.6-35B-A3B-FP8",
        base_url="http://localhost:8000/v1",
    ) == "openai/Qwen/Qwen3.6-35B-A3B-FP8"


def test_missing_openai_credentials_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    message = get_missing_credentials_message("openai/gpt-4.1")

    assert message is not None
    assert "OPENAI_API_KEY" in message


def test_openai_admin_key_satisfies_credential_check(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "test-admin-key")

    assert get_missing_credentials_message("openai/gpt-4.1") is None


def test_explicit_api_key_satisfies_credential_check(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)

    assert get_missing_credentials_message("openai/gpt-4.1", api_key="no-key-required") is None


def test_unknown_provider_skips_credential_check(monkeypatch):
    monkeypatch.delenv("FOO_API_KEY", raising=False)

    assert get_missing_credentials_message("foo/bar-model") is None


def test_load_hermes_provider_config_from_custom_provider(tmp_path):
    config_dir = tmp_path / ".hermes-dev"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "providers:\n"
        "  dgx-qwen:\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        "    api_key: no-key-required\n"
        "    default_model: Qwen/Qwen3.6-35B-A3B-FP8\n"
        "model:\n"
        "  provider: custom:dgx-qwen\n"
        "  default: Qwen/Qwen3.6-35B-A3B-FP8\n"
    )

    provider = load_hermes_provider_config(config_dir)

    assert provider is not None
    assert provider.base_url == "http://127.0.0.1:8000/v1"
    assert provider.api_key == "no-key-required"
    assert provider.model == "openai/Qwen/Qwen3.6-35B-A3B-FP8"
