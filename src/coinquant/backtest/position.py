class Position:
    # 有符号持仓
    position: float = 0.0
    # 持仓均价
    entry_price: float = 0.0
    # 最新标记价格
    mark_price: float = 0.0
    # 浮盈
    unrealized_pnl: float = 0.0
    # 保证金
    margin: float = 0.0
    holding_steps: int = 0
    realized_pnl: float = 0

    total_fee: float = 0