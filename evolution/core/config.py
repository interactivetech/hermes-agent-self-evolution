"""Configuration, provider credential checks, and hermes-agent repo discovery."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class EvolutionConfig:
    """Configuration for a self-evolution optimization run."""

    # hermes-agent repo path
    hermes_agent_path: Path = field(default_factory=lambda: get_hermes_agent_path())

    # Optimization parameters
    iterations: int = 10
    population_size: int = 5

    # LLM configuration
    optimizer_model: str = "openai/gpt-4.1"  # Model for GEPA reflections
    eval_model: str = "openai/gpt-4.1-mini"  # Model for LLM-as-judge scoring
    judge_model: str = "openai/gpt-4.1"  # Model for dataset generation

    # Constraints
    max_skill_size: int = 15_000  # 15KB default
    max_tool_desc_size: int = 500  # chars
    max_param_desc_size: int = 200  # chars
    max_prompt_growth: float = 0.2  # 20% max growth over baseline

    # Eval dataset
    eval_dataset_size: int = 20  # Total examples to generate
    train_ratio: float = 0.5
    val_ratio: float = 0.25
    holdout_ratio: float = 0.25

    # Benchmark gating
    run_pytest: bool = True
    run_tblite: bool = False  # Expensive — opt-in
    tblite_regression_threshold: float = 0.02  # Max 2% regression allowed

    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    create_pr: bool = True
    lm_kwargs: dict = field(default_factory=dict)


_PROVIDER_ENV_VARS = {
    "openai": ("OPENAI_API_KEY", "OPENAI_ADMIN_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "xai": ("XAI_API_KEY",),
}

_KNOWN_PROVIDERS = set(_PROVIDER_ENV_VARS.keys())


@dataclass
class HermesProviderConfig:
    """Resolved provider settings from Hermes config.yaml."""

    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def lm_kwargs(self) -> dict:
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


def get_model_provider(model: str) -> Optional[str]:
    """Extract the LiteLLM provider prefix from a model string."""
    if not model or "/" not in model:
        return None
    provider, _ = model.split("/", 1)
    provider = provider.strip().lower()
    return provider if provider in _KNOWN_PROVIDERS else None


def normalize_model_for_litellm(model: str, base_url: Optional[str] = None) -> str:
    """Prefix raw model ids for OpenAI-compatible endpoints when needed."""
    if not model:
        return model
    if get_model_provider(model):
        return model
    if base_url:
        return f"openai/{model}"
    return model


def get_missing_credentials_message(model: str, api_key: Optional[str] = None) -> Optional[str]:
    """Return a human-friendly credential error if the model's provider is unset."""
    provider = get_model_provider(model)
    if not provider:
        return None

    env_vars = _PROVIDER_ENV_VARS.get(provider)
    if not env_vars:
        return None

    if api_key or any(os.getenv(name) for name in env_vars):
        return None

    env_list = " or ".join(env_vars)
    return (
        f"Model '{model}' requires provider credentials. "
        f"Set {env_list} before running evolution."
    )


def get_hermes_home() -> Path:
    """Discover Hermes home directory."""
    env_home = os.getenv("HERMES_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".hermes"


def load_hermes_provider_config(hermes_home: Optional[Path] = None) -> Optional[HermesProviderConfig]:
    """Load the active provider and model settings from Hermes config.yaml."""
    home = hermes_home or get_hermes_home()
    config_path = home / "config.yaml"
    if not config_path.exists():
        return None

    data = yaml.safe_load(config_path.read_text()) or {}
    model_cfg = data.get("model") or {}
    provider_ref = model_cfg.get("provider")
    default_model = model_cfg.get("default")
    providers = data.get("providers") or {}

    if not provider_ref or not default_model:
        return None

    provider_name = str(provider_ref).split("custom:", 1)[-1]
    provider_cfg = providers.get(provider_name) or {}
    base_url = provider_cfg.get("base_url")
    api_key = provider_cfg.get("api_key")

    model = normalize_model_for_litellm(str(default_model), base_url=base_url)
    return HermesProviderConfig(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def get_hermes_agent_path() -> Path:
    """Discover the hermes-agent repo path.

    Priority:
    1. HERMES_AGENT_REPO env var
    2. ~/.hermes/hermes-agent (standard install location)
    3. ../hermes-agent (sibling directory)
    """
    env_path = os.getenv("HERMES_AGENT_REPO")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    home_path = Path.home() / ".hermes" / "hermes-agent"
    if home_path.exists():
        return home_path

    sibling_path = Path(__file__).parent.parent.parent / "hermes-agent"
    if sibling_path.exists():
        return sibling_path

    raise FileNotFoundError(
        "Cannot find hermes-agent repo. Set HERMES_AGENT_REPO env var "
        "or ensure it exists at ~/.hermes/hermes-agent"
    )
