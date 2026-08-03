from dataclasses import dataclass


@dataclass
class AccountConfig:
    initial_balance: float = 10000.0
    maintenance_margin_rate: float = 0.005

class Account:
    def __init__(self, config: AccountConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self.balance = self.cfg.initial_balance             # 账户余额（结算后）
        self.equity = self.cfg.initial_balance              # 账户权益
        self.available_balance = self.cfg.initial_balance   # 可用余额
        self.used_margin = 0.0                              # 已使用的保证金
        self.maintenance_margin = 0.0                       # 
        # ---------- 盈亏 ----------
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        # ---------- 成本 ----------
        self.total_fee = 0.0
        self.total_funding = 0.0
        # ---------- 风险 ----------
        self.margin_ratio = 0.0
        self.is_liquidated = False

    def update(self, unrealized_pnl: float, margin: float):
        self.unrealized_pnl = unrealized_pnl
        self.used_margin = margin

        self.equity = self.balance + self.unrealized_pnl
        self.available_balance = self.equity - self.used_margin
        self.maintenance_margin = self.used_margin * self.cfg.maintenance_margin_rate

        if self.used_margin > 0.0:
            self.margin_ratio = self.equity / self.used_margin
        else:
            self.margin_ratio = float('inf')

        self.check_liquidation()

    # ----------------------------------------------------
    # 手续费
    # ----------------------------------------------------
    def pay_fee(self, fee: float):
        self.balance -= fee
        self.total_fee += fee

    # ----------------------------------------------------
    # Funding
    # ----------------------------------------------------
    def pay_funding(self, funding: float):
        self.balance -= funding
        self.total_funding += funding

    # ----------------------------------------------------
    # 平仓
    # ----------------------------------------------------
    def realize_pnl(self, pnl: float):
        self.balance += pnl
        self.realized_pnl += pnl

    # ----------------------------------------------------
    # 风险检查
    # ----------------------------------------------------
    def check_liquidation(self):
        self.is_liquidated = self.equity <= self.maintenance_margin
        return self.is_liquidated

    # ----------------------------------------------------
    # Snapshot
    # ----------------------------------------------------
    def snapshot(self):
        return {
            "balance": self.balance,
            "equity": self.equity,
            "available_balance": self.available_balance,
            "used_margin": self.used_margin,
            "maintenance_margin": self.maintenance_margin,
            "margin_ratio": self.margin_ratio,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_fee": self.total_fee,
            "total_funding": self.total_funding,
            "liquidated": self.is_liquidated,
        }