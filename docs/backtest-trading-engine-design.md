# CoinQuant 独立回测交易系统设计

> 状态：设计提案
> 日期：2026-08-03
> 范围：`ExecutionEngine`、`Account`、`Position` 与独立回测编排
> 本文只定义设计，不包含实现代码。

## 1. 系统边界与目标

`Position`、`Account` 和 `ExecutionEngine` 共同构成独立交易模拟内核；`BacktestState` 只负责组织一次回测的运行状态和账本。给定标准化行情与目标合约数量，系统应能独立完成成交、盯市、资金费、强平、期末结算和指标计算，不依赖模型、策略、pandas、PyTorch、数据库或报告模块。

现有 `engine.py` 中的 `BacktestEngine`、旧 `BacktestResult` 及其向量化账务整体废弃，不属于本设计，也不能作为实现或兼容基线。

第一版目标：

1. bar `T` 收盘生成的目标最早在 `T+1` 开盘成交，不使用未来数据。
2. 覆盖开仓、加仓、减仓、平仓和反手，且账务始终守恒。
3. 风险增加前检查保证金，逐 bar 盯市并支持强平。
4. 相同输入与配置产生相同账本和结果。
5. 策略只产生目标，执行内核只处理交易与账务，报告只消费冻结结果。
6. 批量回测与逐 bar 调用使用同一套执行语义。

## 2. 第一版范围与输入

### 2.1 支持范围

- 单账户、单品种、单向净持仓；
- USDT 本位线性永续合约；
- 全仓保证金；
- 市价单立即全部成交；
- 固定手续费率、滑点率和维持保证金率；
- OHLC bar 级撮合，可选资金费率；
- 以目标合约数量执行，而不是累加买卖指令；
- 默认在期末按最后收盘价平仓，可通过配置关闭。

暂不支持多品种共享保证金、逐仓或组合保证金、分档维持保证金、限价单、成交量约束、部分成交、动态 maker/taker 费率、反向或币本位合约、期权、盘口撮合、ADL 和保险基金。

### 2.2 单位与输入契约

`Position.position` 是有符号合约数量：

- 大于 `0`：多仓；
- 小于 `0`：空头；
- 等于 `0`：无持仓。

`MarketBar` 至少包含 `timestamp/open/high/low/close`，可选包含 `funding_rate`。时间戳必须严格递增，所有价格必须是有限正数，且 `low <= min(open, close) <= max(open, close) <= high`。

`TargetInstruction` 至少包含：

- `generated_at`：目标生成时对应的 bar 时间；
- `target_quantity`：有符号目标合约数量；
- `reason` 和 `metadata`：可选审计信息。

`generated_at = T` 的目标只能在 `T+1` 开盘执行。信号值或目标杠杆必须在内核外转换为合约数量：

`目标合约数 = 目标名义杠杆 × 当前权益 / (参考价格 × contract_size)`

## 3. 架构、配置与状态所有权

```text
MarketBar + TargetInstruction
              |
              v
run / advance / submit / finalize
              |
              v
        BacktestState
     /         |          \
Account   ExecutionEngine  Position
     \         |          /
      bar / trade / account ledger
              |
              v
        BacktestResult
```

### 3.1 配置归属

| 配置 | 字段 |
| --- | --- |
| `AccountConfig` | `initial_balance` |
| `PositionConfig` | `leverage`、`contract_size`、`margin_rate` |
| `ExecutionConfig` | `fee_rate`、`slippage_rate`、`quantity_epsilon`、`liquidation_fee_rate` |
| `BacktestConfig` | `force_close_at_end`、`annualization_factor` |

`PositionConfig` 不可变。合约面值、杠杆和维持保证金率共同决定持仓估值，因此不放入 `Account`。未来支持多品种时可以再抽取 `ContractSpec`，但账户仍只接收聚合结果。

### 3.2 状态所有权

- `Position`：持仓数量、均价、标记估值、初始保证金和维持保证金。
- `Account`：余额、账户累计成本、聚合权益和账户级风险状态。
- `ExecutionEngine`：验证并原子地转换 `Position` 与 `Account`，自身只长期持有执行配置。
- `BacktestState`：核心对象、时间游标、待执行目标、运行状态和账本；不复制财务字段。
- 指标和报告：只读取冻结账本，不修改交易状态。

单品种版本中：

```text
Account.unrealized_pnl   == Position.unrealized_pnl
Account.used_margin      == Position.initial_margin
Account.maintenance_margin == Position.maintenance_margin
```

账户字段是持仓估值的聚合结果，不能成为第二套保证金计算来源。

## 4. 核心组件

### 4.1 `Position`

`Position` 是实例级 dataclass，并持有不可变的 `PositionConfig`。

| 字段 | 含义 | 空仓值 |
| --- | --- | --- |
| `position` | 有符号合约数量 | `0` |
| `entry_price` | 当前净持仓均价 | `0` |
| `mark_price` | 最近标记价格 | 最近价格或 `0` |
| `notional` | 按标记价计算的名义价值 | `0` |
| `unrealized_pnl` | 未实现盈亏 | `0` |
| `initial_margin` | 初始保证金 | `0` |
| `maintenance_margin` | 维持保证金 | `0` |
| `holding_steps` | 当前方向连续持有的完整 bar 数 | `0` |

`Position.revalue(mark_price)` 一次性刷新 `mark_price`、`notional`、`unrealized_pnl`、`initial_margin` 和 `maintenance_margin`。它不读取账户余额，也不判断强平。

持仓规则：

- 同方向加仓更新加权均价；
- 减仓不改变剩余仓位均价；
- 全平清空开仓价和估值字段；
- 反手先关闭旧方向，再以本次成交价建立新方向；
- `holding_steps` 每根完整持仓 bar 最多增加一次；
- 累计已实现盈亏和累计费用不保存在 `Position`。

### 4.2 `Account`

| 字段 | 含义 |
| --- | --- |
| `balance` | 已结算余额 |
| `equity` | 余额加未实现盈亏 |
| `available_balance` | 权益减初始保证金占用 |
| `used_margin` | 聚合初始保证金 |
| `maintenance_margin` | 聚合维持保证金 |
| `realized_pnl` | 累计已实现交易盈亏，不扣费用 |
| `unrealized_pnl` | 聚合未实现盈亏 |
| `total_fee` | 累计成交费和强平费 |
| `total_funding` | 累计净支付资金费；支付为正，收取为负 |
| `margin_ratio` | 维持保证金占权益的比例 |
| `is_liquidated` | 是否已触发强平 |

`Account.update` 接收持仓已计算好的未实现盈亏、初始保证金和维持保证金，统一刷新权益、可用余额、风险比率和强平状态。`Account` 不接收杠杆、合约面值或维持保证金率，也不自行计算持仓保证金。

`margin_ratio` 和 `is_liquidated` 属于 `Account`，因为全仓风险以账户总权益覆盖全部持仓维持保证金为准。

### 4.3 `ExecutionEngine`

公开行为：

| 行为 | 结果 |
| --- | --- |
| 执行目标仓位 | 应用滑点、费用和持仓转换，原子更新持仓与账户 |
| 盯市 | 调用 `Position.revalue`，聚合到账户并检查风险 |
| 结算资金费 | 更新余额后重新盯市 |
| 风险强平 | 关闭持仓、收取费用并保持强平标记 |
| 完成 bar | 非空仓的 `holding_steps` 增加一次 |

`ExecutionEngine` 不读取历史数据、不推进时间、不运行模型、不解释信号、不计算报告指标，也不写数据库或报告文件。买卖和各种仓位转换可以有独立路由，但均价、盈亏和费用公式只能有一套实现。

### 4.4 `BacktestState`

| 字段 | 含义 |
| --- | --- |
| `config` | `BacktestConfig` |
| `account`、`position`、`execution` | 本次运行唯一的核心对象 |
| `pending_target` | 上一根收盘后提交、等待本根开盘执行的目标 |
| `current_index`、`last_bar` | 时间游标和最后完成的 bar |
| `status` | `READY/RUNNING/COMPLETED/LIQUIDATED` |
| 三个 ledger | bar、成交和账户事件记录 |

`BacktestState` 只组织对象与记录，不包含成交、盈亏、保证金或强平公式。

## 5. 回测编排

### 5.1 API

```python
def create_backtest_state(
    account_config: AccountConfig,
    position_config: PositionConfig,
    execution_config: ExecutionConfig,
    backtest_config: BacktestConfig,
) -> BacktestState: ...

def advance_bar(state: BacktestState, bar: MarketBar) -> None: ...

def submit_target(
    state: BacktestState,
    target: TargetInstruction,
) -> None: ...

def finalize_backtest(state: BacktestState) -> BacktestResult: ...

def run_backtest(
    state: BacktestState,
    bars: Iterable[MarketBar],
    targets: Mapping[Timestamp, TargetInstruction],
) -> BacktestResult: ...
```

`advance_bar` 是唯一的时间推进入口。调用返回后，上游可以读取收盘快照并调用 `submit_target`；提交只进入 `pending_target`，不立即成交。`run_backtest` 只是组合调用这些函数。

### 5.2 每根 bar 的顺序

1. 校验时间和 OHLC，状态进入 `RUNNING`。
2. 以开盘价盯市；若触发跳空强平，立即强平。
3. 若有资金费率，对开盘前持仓结算资金费并再次检查风险。
4. 在开盘价执行 `pending_target`，执行后清空。
5. 多仓以 `low`、空头以 `high` 做盘中不利价格检查。
6. 未强平时以收盘价盯市。
7. 更新持仓时长，追加账本并提交 `current_index` 和 `last_bar`。

任一风险检查触发强平后，跳过剩余普通事件，只完成强平、记录和游标提交，不再接受新目标。

`advance_bar` 返回后，`submit_target` 只接受 `generated_at == last_bar.timestamp` 的目标。该目标等待下一根开盘，因此动态仓位换算可以使用真实收盘权益而不引入未来函数。

### 5.3 期末处理

`finalize_backtest`：

- 没有已完成 bar 时拒绝运行；
- 已强平时只冻结结果，不重复平仓；
- 默认以 `last_bar.close`、目标数量 `0`、原因 `END_OF_TEST` 普通平仓；
- 不强制平仓时保留期末未实现盈亏与未平仓数量；
- 丢弃并记录最后一根收盘生成的未执行目标；
- 正常结束后状态为 `COMPLETED`。

`COMPLETED` 或 `LIQUIDATED` 状态不能继续推进或提交目标。重复 finalize 不修改状态并返回等值结果。

## 6. 成交状态机与原子性

令 `q0` 为执行前持仓，`qt` 为目标持仓，`dq = qt - q0`。`dq > 0` 为买入，`dq < 0` 为卖出。

当 `abs(dq) <= quantity_epsilon` 时不成交、不收费，只完成盯市。

| 条件 | 转换 |
| --- | --- |
| `q0 == 0` | 开仓 |
| `q0 × dq > 0` | 同方向加仓 |
| `q0 × dq < 0` 且 `abs(dq) < abs(q0)` | 减仓 |
| `q0 × dq < 0` 且数量相等 | 全平 |
| `q0 × dq < 0` 且 `abs(dq) > abs(q0)` | 反手 |

反手按账务拆成“关闭旧仓”和“建立新仓”，但可表现为一笔成交：全部数量按同一成交价收费，只有关闭数量产生已实现盈亏，新方向均价为该成交价。

一次执行必须先在临时状态完成以下步骤，再原子提交：

1. 校验输入和账户状态；
2. 用当前标记价格刷新旧状态；
3. 计算成交价、持仓转换、盈亏、费用和投影保证金；
4. 对风险增加部分检查可用余额和强平条件；
5. 成功时同时提交 `Position` 与 `Account`，失败时不留下任何写入；
6. 再次盯市并返回不可变执行结果。

保证金不足时：

| 请求 | 处理 |
| --- | --- |
| 减仓或平仓 | 始终允许 |
| 开仓或同方向加仓 | 整个风险增加部分拒绝 |
| 反手 | 先平旧仓；新方向不足则停在空仓并记录降级原因 |

反手降级是一次明确记录的成功业务结果，关闭旧仓和拒绝新仓必须在同一次原子提交中完成。不得静默缩小目标。风险增加后的投影必须满足 `available_balance >= 0` 和 `equity > maintenance_margin`。

## 7. 统一公式与风险

以下公式适用于 USDT 本位线性合约；`contract_size`、`leverage` 和 `margin_rate` 来自 `PositionConfig`。

### 7.1 成交与持仓

```text
买入成交价 = reference_price × (1 + slippage_rate)
卖出成交价 = reference_price × (1 - slippage_rate)

trade_notional = abs(dq) × fill_price × contract_size
fee = trade_notional × fee_rate

同方向加仓均价 =
    (abs(q0) × old_entry + abs(dq) × fill_price)
    / abs(q0 + dq)

close_qty = min(abs(q0), abs(dq))，仅在 q0 × dq < 0 时非零
realized_pnl =
    close_qty × sign(q0) × (fill_price - entry_price) × contract_size

unrealized_pnl =
    position × (mark_price - entry_price) × contract_size

notional = abs(position) × mark_price × contract_size
initial_margin = notional / leverage
maintenance_margin = notional × margin_rate

funding_payment =
    position × mark_price × contract_size × funding_rate
```

手续费不混入 `realized_pnl`；`total_fee` 包含普通成交费和强平费。正资金费率时多仓支付、空头收取。

### 7.2 账户与风险

```text
balance =
    initial_balance + realized_pnl - total_fee - total_funding

equity = balance + unrealized_pnl
available_balance = equity - used_margin
```

`margin_ratio` 定义为：

```text
if maintenance_margin == 0:
    margin_ratio = 0
elif equity <= 0:
    margin_ratio = +inf
else:
    margin_ratio = maintenance_margin / equity
```

强平的权威条件是：

`maintenance_margin > 0 and equity <= maintenance_margin`

`margin_ratio` 随风险上升而增大；权益为正时达到 `1` 即进入强平区。`is_liquidated` 使用粘性更新：

`is_liquidated = is_liquidated or liquidation_trigger`

bar 内多仓使用 `low`、空头使用 `high` 检查强平。触发后按不利价格叠加强平方向滑点关闭全部仓位并收取费用，余额允许为负且不截断。OHLC 无法还原盘中路径，因此这是保守的 bar 级近似。

## 8. 结果、账本与指标

每次执行或风险事件返回不可变结果，至少记录时间、事件类型、请求及实际数量、成交价格和名义价值、已实现盈亏、费用、拒绝或降级原因，以及事件后的账户和持仓快照。

| 账本 | 内容 |
| --- | --- |
| Bar ledger | OHLC、目标、执行结果、收盘持仓和权益 |
| Trade ledger | 实际成交、费用、已实现盈亏和原因 |
| Account ledger | 关键事件后的账户与持仓快照 |

`BacktestResult` 是不可变结果，包含运行状态、时间范围、最终快照、三个冻结账本、配置快照和 `BacktestMetrics`。

第一版指标：

| 指标 | 口径 |
| --- | --- |
| `total_return` | `final_equity / initial_balance - 1` |
| `annualized_return` | `(final_equity / initial_balance) ** (annualization_factor / bar_count) - 1` |
| `annualized_volatility` | `std(bar_returns, ddof=1) × sqrt(annualization_factor)` |
| `max_drawdown` | 权益相对历史最高权益的最小跌幅 |
| `sharpe` | 无风险利率为 `0`；收益标准差为 `0` 时为空 |
| 成交与成本 | 从 trade ledger 和最终账户累计项汇总 |

权益序列以初始资金为第一个点，之后使用每根 bar 的收盘权益；期末强制平仓后的最终权益替换最后一个点。年化收益要求最终权益为正；波动率和 Sharpe 要求至少两个有效 bar 收益。总收益和最大回撤保留负权益结果。

## 9. 不变量与边界规则

### 9.1 核心不变量

每次事件提交后必须满足：

1. `equity == balance + unrealized_pnl`。
2. `balance == initial_balance + realized_pnl - total_fee - total_funding`。
3. 账户的未实现盈亏、初始保证金和维持保证金与持仓估值一致。
4. 空仓时开仓价、名义价值、持仓未实现盈亏、两类保证金和持仓时长为零。
5. 名义价值、费用、保证金和风险比率非负。
6. 减仓不改变剩余均价；同方向加仓不产生已实现盈亏；目标不变不收费。
7. 无维持保证金时 `margin_ratio == 0`；有维持保证金且权益非正时为正无穷。
8. `is_liquidated` 在 reset 前不能恢复为假。
9. `pending_target.generated_at == last_bar.timestamp`，且执行时间晚于生成时间。
10. 账本只追加不可变快照；批量与逐 bar 调用产生等值结果。

金额和数量使用统一浮点容差，不使用裸 `==`。

### 9.2 异常与业务拒绝

以下情况属于输入或配置错误，直接抛异常：

- 非有限配置值、数量或价格，非正价格，非法 OHLC；
- 非正初始资金、杠杆、合约面值或数量容差；
- 费率、滑点率、维持保证金率或强平费率不在 `[0, 1)`；
- 年化因子不为空且非正；
- 时间倒序或重复；
- 目标时间无法对齐、目标重复或目标数量非有限；
- 无 bar 时 finalize。

保证金不足、目标只能部分完成等属于业务结果，必须记录原因，不作为内部异常。目标差小于数量容差视为无成交。已强平账户只允许读取快照或 reset。

## 10. 文件边界与迁移

| 文件 | 责任 |
| --- | --- |
| `backtest/account.py` | `AccountConfig`、`Account`、账户快照 |
| `backtest/position.py` | `PositionConfig`、`Position`、持仓估值与快照 |
| `backtest/execution.py` | `ExecutionConfig`、`ExecutionEngine`、执行结果 |
| `backtest/simulation.py` | 标准输入、`BacktestState`、编排函数与新 `BacktestResult` |
| `backtest/metrics.py` | 从冻结账本计算指标 |
| `backtest/strategy.py`、`backtest/report.py` | 可选上游适配与下游展示 |

迁移顺序：

1. 将 `Position` 改为实例 dataclass，新增 `PositionConfig`、`notional`、`initial_margin` 和 `maintenance_margin`。
2. 删除 `Account.update` 内部的保证金公式，改为聚合持仓估值，并修正 `margin_ratio` 与强平粘性。
3. 将 `ExecutionEngine` 移到 `execution.py`，实现原子成交状态机。
4. 在 `simulation.py` 实现输入值对象、`BacktestState` 和编排函数。
5. 改造策略、handler、指标和报告以使用新结果。
6. 删除 `BacktestEngine`、旧 `BacktestResult` 和旧向量化账务，清理全部旧导入。

新实现不得导入旧 `engine.py`，也不得用旧输出作为测试真值。

## 11. 验证要求

| 层级 | 必须覆盖 |
| --- | --- |
| 公式单元测试 | 多空盈亏、加权均价、滑点、费用、资金费、两类保证金和 `margin_ratio` 边界 |
| 状态转换测试 | 开仓、加仓、减仓、全平、双向反手、拒绝和反手降级 |
| 属性测试 | 账户恒等式、拆单一致性、多空对称、同输入可复现 |
| 编排集成测试 | `T+1` 成交、跳空与盘中强平、期末处理、最后目标不成交 |
| 对账测试 | 三个账本可互相重算，指标只依赖冻结结果 |
| 独立性测试 | 只用内存 bar 和目标即可运行，核心不导入模型、pandas、PyTorch、报告或旧 `engine.py` |

完成标准是上述测试全部通过、每个事件后不变量成立、保证金不足不留下部分写入，并且 `run_backtest` 与“逐 bar + submit + finalize”产生相同结果。
