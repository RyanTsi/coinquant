from coinquant.backtest.alpha_backtester import (
    AlphaBacktester,
    AlphaThresholds,
    BacktestConfig,
    BacktestResult,
    BacktestSplit,
)
from coinquant.backtest.alpha_grid_search import (
    AlphaGridSearcher,
    AlphaGridSearchConfig,
    AlphaGridSearchResult,
    GridSortKey,
)
from coinquant.config import settings


_MISSING = object()


def _get_setting(section, key: str, default=_MISSING):
    if section is None:
        return default
    if hasattr(section, "get"):
        return section.get(key, default)
    return getattr(section, key, default)


def _first_setting(sections, key: str, default):
    for section in sections:
        value = _get_setting(section, key, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _backtest_settings():
    return _get_setting(settings, "backtest", {})


def _execution_kwargs(*sections) -> dict:
    return {
        "initial_cash": float(_first_setting(sections, "initial_cash", 10_000.0)),
        "cash_fraction": float(_first_setting(sections, "cash_fraction", 1.0 / 3.0)),
        "fee_rate": float(_first_setting(sections, "fee_rate", 0.0004)),
        "min_trade_cash": float(_first_setting(sections, "min_trade_cash", 10.0)),
        "max_buy_count": int(_first_setting(sections, "max_buy_count", 3)),
        "min_hold_bars": int(_first_setting(sections, "min_hold_bars", 0)),
        "cooldown_bars": int(_first_setting(sections, "cooldown_bars", 0)),
        "liquidate_on_end": bool(_first_setting(sections, "liquidate_on_end", True)),
    }


def _optional_int(value):
    return None if value is None else int(value)


def _optional_float(value):
    return None if value is None else float(value)


def _backtest_split(value) -> BacktestSplit:
    if isinstance(value, BacktestSplit):
        return value
    return BacktestSplit(str(value))


def _grid_sort_key(value) -> GridSortKey:
    if isinstance(value, GridSortKey):
        return value
    return GridSortKey(str(value))


def run_alpha_backtest(
    symbol: str | None = None,
    period: str | None = None,
    split: BacktestSplit | str | None = None,
) -> BacktestResult:
    backtest_config = _backtest_settings()
    thresholds = _get_setting(backtest_config, "thresholds", {})
    execution = _get_setting(backtest_config, "execution", {})
    config = BacktestConfig(
        symbol=symbol or str(_get_setting(backtest_config, "symbol", "BTC/USDT")),
        period=period or str(_get_setting(backtest_config, "period", "15m")),
        split=_backtest_split(split or _get_setting(backtest_config, "split", BacktestSplit.test.value)),
        thresholds=AlphaThresholds(
            long_entry=float(_get_setting(thresholds, "long_entry", 0.0)),
            short_entry=float(_get_setting(thresholds, "short_entry", 0.0)),
            long_exit=float(_get_setting(thresholds, "long_exit", 0.0)),
            short_exit=float(_get_setting(thresholds, "short_exit", 0.0)),
        ),
        **_execution_kwargs(execution),
        output_dir=_get_setting(backtest_config, "output_dir", None),
    )
    return AlphaBacktester(config).run()


def run_alpha_grid_search(
    symbol: str | None = None,
    period: str | None = None,
    split: BacktestSplit | str | None = None,
) -> AlphaGridSearchResult:
    backtest_settings = _backtest_settings()
    grid_settings = _get_setting(backtest_settings, "grid_search", {})
    execution = _get_setting(backtest_settings, "execution", {})
    grid_execution = _get_setting(grid_settings, "execution", {})
    output_dir = _get_setting(grid_settings, "output_dir", _MISSING)
    if output_dir is _MISSING or output_dir is None:
        output_dir = _get_setting(backtest_settings, "output_dir", None)
    backtest_config = BacktestConfig(
        symbol=symbol
        or str(_get_setting(grid_settings, "symbol", _get_setting(backtest_settings, "symbol", "BTC/USDT"))),
        period=period
        or str(_get_setting(grid_settings, "period", _get_setting(backtest_settings, "period", "15m"))),
        split=_backtest_split(split or _get_setting(grid_settings, "split", BacktestSplit.valid.value)),
        thresholds=AlphaThresholds(
            long_entry=0.0,
            short_entry=0.0,
            long_exit=0.0,
            short_exit=0.0,
        ),
        **_execution_kwargs(grid_execution, execution),
        output_dir=output_dir,
    )
    search_config = AlphaGridSearchConfig(
        backtest_config=backtest_config,
        grid_size=int(_get_setting(grid_settings, "grid_size", 4)),
        entry_min_quantile=float(_get_setting(grid_settings, "entry_min_quantile", 0.70)),
        entry_max_quantile=float(_get_setting(grid_settings, "entry_max_quantile", 0.95)),
        exit_min_quantile=float(_get_setting(grid_settings, "exit_min_quantile", 0.01)),
        exit_max_quantile=float(_get_setting(grid_settings, "exit_max_quantile", 0.30)),
        top_k=int(_get_setting(grid_settings, "top_k", 20)),
        min_trades=int(_get_setting(grid_settings, "min_trades", 1)),
        max_trades=_optional_int(_get_setting(grid_settings, "max_trades", None)),
        max_turnover=_optional_float(_get_setting(grid_settings, "max_turnover", None)),
        sort_by=_grid_sort_key(_get_setting(grid_settings, "sort_by", GridSortKey.score.value)),
        drawdown_weight=float(_get_setting(grid_settings, "drawdown_weight", 0.5)),
        turnover_weight=float(_get_setting(grid_settings, "turnover_weight", 0.0001)),
        trade_weight=float(_get_setting(grid_settings, "trade_weight", 0.0001)),
        output_dir=output_dir,
    )
    return AlphaGridSearcher(search_config).run()
