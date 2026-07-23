"""Lightweight registry for the official When2Tool environment classes."""

from importlib import import_module


ENV_REGISTRY = {
    "CalculatorEnv": ("calculator_env", "CalculatorEnv"),
    "RetrieverEnv": ("retriever_env", "RetrieverEnv"),
    "ListManipulationEnv": ("list_manipulation_env", "ListManipulationEnv"),
    "StatisticsEnv": ("statistics_env", "StatisticsEnv"),
    "CountingEnv": ("counting_env", "CountingEnv"),
    "MatrixEnv": ("matrix_env", "MatrixEnv"),
    "PrimeEnv": ("prime_env", "PrimeEnv"),
    "HistoricalYearEnv": ("historical_year_env", "HistoricalYearEnv"),
    "GameRuleEnv": ("game_rule_env", "GameRuleEnv"),
    "HashEnv": ("hash_env", "HashEnv"),
    "DecodingEnv": ("decoding_env", "DecodingEnv"),
    "DateTimeEnv": ("datetime_env", "DateTimeEnv"),
    "CodeExecutorEnv": ("code_executor_env", "CodeExecutorEnv"),
    "ScheduleEnv": ("schedule_env", "ScheduleEnv"),
    "RegexMatchEnv": ("regex_match_env", "RegexMatchEnv"),
}


def load_env_class(env_name):
    if env_name not in ENV_REGISTRY:
        raise KeyError(env_name)
    module_name, class_name = ENV_REGISTRY[env_name]
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, class_name)


def __getattr__(name):
    if name in ENV_REGISTRY:
        cls = load_env_class(name)
        globals()[name] = cls
        return cls
    raise AttributeError(name)


__all__ = list(ENV_REGISTRY) + ["ENV_REGISTRY", "load_env_class"]
