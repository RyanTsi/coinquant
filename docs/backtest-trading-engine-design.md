# CoinQuant 回测交易引擎设计

> 状态：设计提案
> 日期：2026-08-03
> 范围：`ExecutionEngine`、`Account`、`Position` 以及独立回测编排状态
> 本文只定义设计，不包含实现代码。

## 1. 背景

当前 `src/coinquant/backtest` 下已经存在三块基础能力：

- `ExecutionEngine` 已有目标仓位执行的轮廓，但买卖、加减仓、反手、盯市及费用计算尚未实现。
- `Account` 已保存余额、权益、保证金、盈亏、费用和强平状态。
- `Position` 已保存有符号持仓、开仓均价、标记价格、盈亏、保证金和持仓时长。

这三者需要共同组成一个不依赖模型、策略、pandas、PyTorch 和报告模块的独立交易模拟内核，并可靠表达以下交易细节：

- 加仓、减仓、平仓和反手时的不同账务变化；
- 成交价滑点、手续费、资金费；
- 杠杆、初始保证金、维持保证金和强平；
- 订单被拒绝、部分完成目标仓位及强制平仓；
- 每笔成交、每根 K 线和每次账户变化的可审计记录；
- 账户收益指标，包括但不限于收益率、波动率和回撤。

因此需要以 `ExecutionEngine` 作为状态转换入口，以 `Position` 保存持仓事实，以 `Account` 保存资金事实，并新建 `BacktestState` 聚合一次回测所需的三者、时间状态和账本。回测由独立的编排函数推进，输入仅为标准化行情和目标合约数量序列。

现有 `engine.py` 中的 `BacktestEngine` 及其向量化收益计算属于废弃代码，不是本设计的组成部分，也不能作为实现参考。模型推理和 fast/slow 策略只能在交易模拟内核之外把自身输出适配为标准目标序列。

## 2. 设计目标

第一版系统需要满足：

1. **无未来函数**：在 K 线 `T` 收盘后产生的目标仓位，最早只能在 `T+1` 开盘成交。
2. **账务守恒**：任意时刻都能从初始资金、已实现盈亏、手续费、资金费和未实现盈亏解释账户权益。
3. **转换完整**：覆盖空仓、做多、做空之间的开仓、加仓、减仓、平仓和反手。
4. **风险真实**：风险增加前校验可用余额，逐 bar 盯市并支持维持保证金强平。
5. **结果可复现**：同样的数据、配置和目标仓位序列必须产生同样的成交和权益结果。
6. **职责清晰**：策略只决定目标风险，执行层只处理成交与账务，报告层只消费事件记录。
7. **便于扩展**：第一版保持单品种简单模型，但数据结构和事件接口不阻碍后续多品种、限价单和不同合约类型。
8. **独立可运行**：只提供标准化 bar 和目标合约数量，无需加载任何模型或旧回测类即可完成回测并产出账本和指标。

## 3. 第一版范围和假设

### 3.1 支持范围

- 单账户、单品种；
- USDT 本位线性永续合约；
- 单向持仓模式（同一品种只有一个净头寸）；
- 全仓保证金语义；
- 市价单、立即全部成交；
- 固定比例手续费和固定比例滑点；
- OHLC bar 级撮合；
- 可选资金费率输入；
- 目标仓位执行，而不是买卖指令累加；
- 回测结束时可配置是否强制平仓，默认强制平仓。

### 3.2 暂不支持

- 多品种共享保证金；
- 逐仓、组合保证金和分档维持保证金；
- 限价单、挂单队列、成交量约束和部分成交；
- maker/taker 动态费率；
- 反向合约、币本位合约和期权；
- 盘口级撮合；
- 自动减仓（ADL）和保险基金。

这些能力应在核心账务不变量稳定后再扩展。

## 4. 核心设计决策

### 4.1 合约与计价语义

第一版使用线性合约，所有资金和盈亏均以报价币 USDT 计价。

- `PositionConfig.contract_size`：每张合约对应的基础资产数量；默认 `1.0`。
- `Position.position`：**有符号合约数量**，不是信号值，也不是账户资金比例。
- `position > 0` 表示多仓，`position < 0` 表示空头仓位，`position == 0` 表示空仓。
- 所有价格必须是有限正数，所有数量必须是有限数。

明确数量单位非常重要。现有策略产生的 `-1/0/1` 以及 RL 环境产生的目标杠杆不能直接由执行引擎猜测含义，而应先经过仓位换算器：

`目标合约数 = 目标名义杠杆 × 当前权益 / (参考价格 × contract_size)`

例如，目标名义杠杆为 `-1.0` 表示建立相当于账户权益 1 倍名义价值的空头仓位。仓位换算器还应负责交易步长和最大杠杆裁剪。这样，策略与执行层可以各自保持单一语义。

### 4.2 配置的唯一来源

目标设计中：

| 配置归属 | 字段 | 含义 |
| --- | --- | --- |
| `AccountConfig` | `initial_balance` | 初始报价币余额 |
| `PositionConfig` | `leverage` | 当前持仓使用的杠杆 |
| `PositionConfig` | `contract_size` | 单张合约的基础资产数量 |
| `PositionConfig` | `maintenance_margin_rate` | 持仓维持保证金率 |
| `ExecutionConfig` | `fee_rate` | 单边成交手续费率 |
| `ExecutionConfig` | `slippage_rate` | 相对参考价格的单边滑点率 |
| `ExecutionConfig` | `quantity_epsilon` | 判断数量为零的容差 |
| `ExecutionConfig` | `liquidation_fee_rate` | 可选强平附加费率 |
| `BacktestConfig` | `force_close_at_end` | 期末是否强制平仓 |
| `BacktestConfig` | `annualization_factor` | 收益和波动率年化因子；不年化时可为空 |

第一版把合约面值、杠杆和维持保证金率集中在不可变的 `PositionConfig`，因为它们共同决定单个持仓的估值与保证金。未来支持多品种或分档保证金时，可以再把合约静态参数抽成 `ContractSpec`，但 `Account` 仍只接收聚合结果。

### 4.3 状态所有权

- `Position` 是持仓数量、持仓均价、标记估值、初始保证金和维持保证金的唯一持仓级事实来源。
- `Account` 是余额、账户级累计成本和聚合风险状态的唯一事实来源；它不持有杠杆参数，也不从初始保证金反推维持保证金。
- `ExecutionEngine` 不长期持有账户副本；它负责验证输入、计算一次状态转换并原子地提交到传入的 `Position` 和 `Account`。
- `BacktestState` 只聚合上述三个对象、待执行目标、时间游标、运行状态和账本，不复制余额、仓位或盈亏字段。
- 回测编排函数只调用三者公开行为，不直接给 `Account` 或 `Position` 的财务字段赋值。
- 报告与指标只能读取快照和账本，不得反向改变交易状态。

在单品种版本中，`Account.unrealized_pnl`、`Account.used_margin`、`Account.maintenance_margin` 分别等于 `Position.unrealized_pnl`、`Position.initial_margin`、`Position.maintenance_margin`。账户字段是从持仓估值结果聚合得到的只读汇总，不能成为第二套计算来源。

## 5. 独立回测系统与编排结构

### 5.1 系统边界

独立回测系统的最小输入是标准化的 `MarketBar` 序列和按 bar 收盘生成的 `TargetInstruction` 序列，最小输出是新的 `BacktestResult`。策略名称、预测值等信息只能作为可选元数据透传，不参与成交或账务公式。

```text
MarketBar + TargetInstruction
              |
              v
run / advance / submit / finalize
      （时间与事件编排函数）
              |
              v
        BacktestState
     /         |          \
Account   ExecutionEngine  Position
     \         |          /
      bar / trade / account ledger
              |
              v
        BacktestResult + Metrics
```

这个边界保证可以直接构造行情和目标数量做测试，也可以由任意模型、人工规则或 RL 环境生成相同的数据契约后复用。编排层不得导入模型注册表、数据集构建器、数据库、pandas、PyTorch 或 HTML 报告。

### 5.2 新编排数据结构：`BacktestState`

`BacktestState` 是一次运行的聚合状态，不是另一个交易引擎。建议字段如下：

| 字段 | 类型/语义 |
| --- | --- |
| `config` | `BacktestConfig`，只含编排级配置 |
| `account` | 本次运行唯一的 `Account` 实例 |
| `position` | 本次运行唯一的 `Position` 实例 |
| `execution` | 本次运行使用的 `ExecutionEngine` 实例 |
| `pending_target` | 上一根 bar 收盘生成、等待本根开盘执行的目标；初始为空 |
| `current_index` | 当前 bar 序号；开始前为 `-1` |
| `last_bar` | 最近已完成的 `MarketBar`；用于校验时间递增和期末结算，初始为空 |
| `status` | `READY/RUNNING/COMPLETED/LIQUIDATED/FAILED` |
| `bar_ledger` | 每根已完成 bar 的不可变记录 |
| `trade_ledger` | `ExecutionEngine` 返回的全部实际成交记录 |
| `account_ledger` | 每个关键事件后的账户与持仓快照 |

`BacktestState` 可以提供追加记录、状态迁移和 `to_result()` 等轻量方法，但不能包含成交价、盈亏、费用、保证金或强平公式。这些公式只能存在于 `ExecutionEngine`、`Account` 和 `Position` 的公开行为中。

### 5.3 标准输入契约

`MarketBar` 至少包含 `timestamp/open/high/low/close`，可选包含 `funding_rate`。`TargetInstruction` 至少包含：

- `generated_at`：目标在哪根 bar 收盘后生成；
- `target_quantity`：有符号目标合约数量，与 `Position.position` 单位完全一致；
- `metadata`：可选的信号、模型名或追踪信息，内核只透传；
- `reason`：可选的策略、期末平仓或人工干预原因。

目标输入必须按 `generated_at` 对齐行情。`generated_at = T` 的目标只能进入 `pending_target`，并在存在 `T+1` 时于下一根开盘执行。编排层不接收含糊的 `-1/0/+1` 信号；信号或目标名义杠杆必须在系统外先换算为合约数量。

### 5.4 编排 API

推荐使用模块级函数组织回测，避免再创建一个承担模型、数据和交易逻辑的 `*Engine`：

```python
def create_backtest_state(
    account_config: AccountConfig,
    position_config: PositionConfig,
    execution_config: ExecutionConfig,
    backtest_config: BacktestConfig,
) -> BacktestState: ...

def advance_bar(
    state: BacktestState,
    bar: MarketBar,
) -> None: ...

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

`advance_bar` 是唯一的时间推进入口，便于逐步调试和 RL 复用。调用返回后，上游可以读取收盘快照，完成仓位换算，再用 `submit_target` 将目标放入队列；`submit_target` 只接受 `generated_at == state.last_bar.timestamp` 的目标且不立即成交。`finalize_backtest` 使用 `state.last_bar.close` 执行期末处理并冻结结果；`run_backtest` 只负责校验完整输入，对每根 bar 依次调用 `advance_bar` 和可选的 `submit_target`，最后调用 `finalize_backtest`。这些函数都必须通过 `state.execution` 修改交易和账务状态。

### 5.5 文件边界

| 文件 | 责任 |
| --- | --- |
| `backtest/account.py` | `AccountConfig`、`Account`、账户快照 |
| `backtest/position.py` | `PositionConfig`、`Position`、持仓估值和持仓快照 |
| `backtest/execution.py` | `ExecutionEngine`、执行配置、执行结果和成交记录 |
| `backtest/simulation.py` | `MarketBar`、`TargetInstruction`、`BacktestState`、编排函数和新 `BacktestResult` |
| `backtest/metrics.py` | 只从冻结账本计算收益、波动率、回撤等指标 |
| `backtest/strategy.py` | 可选适配层；把具体策略输出换算为 `TargetInstruction` |
| `backtest/report.py` | 可选展示层；只消费新的 `BacktestResult` |

`engine.py` 中现有的 `BacktestEngine`、旧 `BacktestResult` 及其专用辅助函数不迁入上述核心文件。迁移完成后删除；迁移期间也不允许新模块依赖它们。

## 6. 类职责与数据模型

### 6.1 `Position`

`Position` 应是实例级 dataclass，并持有不可变的 `PositionConfig`；禁止依赖当前的类属性作为可变状态。建议字段语义如下：

| 字段 | 含义 | 空仓值 |
| --- | --- | --- |
| `position` | 有符号合约数量 | `0.0` |
| `entry_price` | 当前净持仓的成交加权均价 | `0.0` |
| `mark_price` | 最近一次有效标记价格 | 最近价格或 `0.0` |
| `notional` | 按标记价计算的持仓名义价值 | `0.0` |
| `unrealized_pnl` | 按标记价计算的未实现盈亏 | `0.0` |
| `initial_margin` | `notional / leverage`，当前初始保证金占用 | `0.0` |
| `maintenance_margin` | `notional × maintenance_margin_rate`，当前维持保证金 | `0.0` |
| `holding_steps` | 当前方向连续持有的完整 bar 数 | `0` |
约束：

- 空仓时必须令 `entry_price`、`notional`、`unrealized_pnl`、`initial_margin`、`maintenance_margin` 和 `holding_steps` 为零；
- 减仓但未平仓时保持原 `entry_price`；
- 同方向加仓时更新加权均价；
- 反手时先结算旧方向，剩余数量以本次成交价作为新方向的 `entry_price`；
- `holding_steps` 只能由 bar 推进事件增加，不能因同一 bar 内多次盯市增加；
- 累计已实现盈亏和累计费用只存于 `Account`，逐笔值存于账本，不在 `Position` 中保留重复累计字段。

`Position.revalue(mark_price)` 使用自身配置一次性刷新 `mark_price`、`notional`、`unrealized_pnl`、`initial_margin` 和 `maintenance_margin`。它只处理持仓级估值，不读取账户余额，也不判断账户是否强平。

### 6.2 `Account`

`Account` 负责账户资金和风险汇总，不负责判断成交属于加仓还是减仓。

| 字段 | 含义 |
| --- | --- |
| `balance` | 已结算余额，已包含已实现盈亏、手续费和资金费 |
| `equity` | 余额加全部未实现盈亏 |
| `available_balance` | 权益减初始保证金占用 |
| `used_margin` | 全部持仓的初始保证金之和 |
| `maintenance_margin` | 全部持仓的维持保证金之和 |
| `realized_pnl` | 累计已实现交易盈亏，不扣费用 |
| `unrealized_pnl` | 全部持仓未实现盈亏之和 |
| `total_fee` | 累计支付的成交手续费，非负 |
| `total_funding` | 累计净支付资金费；支付为正，收取为负 |
| `margin_ratio` | `maintenance_margin / equity`；风险占用越高数值越大 |
| `is_liquidated` | 是否已经触发强平 |

`Account.update` 应演化为一次完整的账户刷新：接收 `Position` 已计算好的未实现盈亏、初始保证金和维持保证金，聚合后统一计算权益、可用余额、`margin_ratio` 和风险状态。`Account` 不接收杠杆或合约面值，也不自行计算持仓保证金。不能在手续费已扣除但权益尚未刷新时向外暴露半更新状态。

`margin_ratio` 和 `is_liquidated` 仍属于 `Account`：第一版虽然只有一个持仓，但全仓保证金的风险判断以账户总权益覆盖全部持仓维持保证金为准；未来扩展多品种时只需改变聚合输入，不需要把账户风险再从 `Position` 迁回账户。

### 6.3 `ExecutionEngine`

`ExecutionEngine` 负责：

- 将目标合约数量与当前数量之差转换为买入或卖出成交；
- 对参考价格施加有方向的滑点；
- 识别开仓、加仓、减仓、平仓和反手；
- 路由持仓转换、计算已实现盈亏和手续费，并调用 `Position.revalue` 完成持仓估值；
- 在风险增加前执行余额校验；
- 以一次原子状态转换更新 `Position` 和 `Account`；
- 生成不可变的执行结果和成交记录；
- 处理盯市、资金费和强平。

`ExecutionEngine` 不负责：

- 读取历史数据、选择下一根 bar 或持有时间游标；
- 运行模型和生成信号；
- 把信号值解释成合约数量；
- 计算 Sharpe、回撤等报告指标；
- 保存 HTML 或数据库文件。

建议对外行为契约：

| 行为 | 输入 | 输出/副作用 |
| --- | --- | --- |
| 执行目标仓位 | 目标数量、参考价格、标记价格、时间、持仓、账户 | 执行结果；原子更新持仓和账户 |
| 盯市 | 标记价格、持仓、账户 | 先调用 `Position.revalue`，再把估值汇总到 `Account` 并检查风险 |
| 结算资金费 | 资金费率、标记价格、持仓、账户 | 资金费事件；更新账户余额后重新盯市 |
| 风险强平 | 清算参考价格、持仓、账户 | 强平成交；账户标记为已强平 |
| 完成 bar | 持仓、bar 序号或时间 | 非空仓的 `holding_steps` 只增加一次 |
| 创建快照 | 时间、持仓、账户 | 不可变持仓/账户快照，无状态修改 |

当前轮廓中的 `buy`、`sell` 适合作为方向路由。`_increase_long` 等六个方法可以保留为可读的状态分派入口，但均价和已实现盈亏公式应集中在一个通用的“应用成交”过程，避免多处公式产生不一致。

### 6.4 执行结果、回测结果与账本

每次调用执行引擎必须返回不可变的 `ExecutionResult`，至少包含：

- 时间、请求目标数量、执行前数量、执行后数量；
- 参考价格、成交方向、成交数量、成交价格、成交名义价值；
- 已实现盈亏、手续费、资金费；
- 请求是否完全完成；
- 拒绝或降级原因；
- 是否触发强平；
- 执行后的 `Position` 和 `Account` 快照。

即使没有成交，盯市、资金费、拒绝和强平也必须返回可区分的事件结果。编排层只追加结果，不能根据结果重新计算或覆盖账户字段。

回测结果维护三个相互关联的账本：

1. **Bar ledger**：每根 K 线一行，记录 OHLC、目标元数据、请求及实际目标、拒绝原因、收盘持仓和权益。
2. **Trade ledger**：每次实际成交一行，记录成交、费用、已实现盈亏和交易原因。
3. **Account ledger**：每次关键事件后的账户快照，用于对账和定位强平。

报告指标应优先从账本派生，而不是在执行过程中维护多个容易漂移的累计指标。

新的 `BacktestResult` 是一次运行结束后的不可变值对象，不复用旧 `BacktestEngine` 的同名类型。至少包含：

- 运行状态、起止时间、bar 数量和最终 `AccountSnapshot`、`PositionSnapshot`；
- 冻结后的 bar、trade、account 三个账本；
- 由账本计算的 `BacktestMetrics`；
- 使用的 `AccountConfig`、`PositionConfig`、`ExecutionConfig`、`BacktestConfig` 快照，保证结果可复现。

`BacktestMetrics` 第一版至少包含：

| 指标 | 数据来源与口径 |
| --- | --- |
| `total_return` | `final_equity / initial_balance - 1` |
| `annualized_return` | `(final_equity / initial_balance) ** (annualization_factor / bar_count) - 1` |
| `annualized_volatility` | `std(bar_returns, ddof=1) × sqrt(annualization_factor)` |
| `max_drawdown` | 权益相对历史最高权益的最小跌幅；最高权益序列包含初始资金点 |
| `sharpe` | 无风险利率默认零；按相同年化因子计算，标准差为零时为空 |
| `trade_count` | `trade_ledger` 中实际成交数量，不含零成交事件 |
| `total_fee/total_funding` | 来自最终账户累计项，并与账本汇总对账 |
| `liquidated` | 最终运行状态是否为 `LIQUIDATED` |

指标函数只接收冻结账本和初始资金。执行过程中不得在 `Account` 中累计回撤、收益率或波动率；`Account` 只维护当前及累计账务事实。

指标的权益时间序列以初始资金为第一个点，之后使用每根 bar 的收盘权益。如果 `finalize_backtest` 在最后收盘执行强制平仓，则最后一个权益点使用期末结算后的最终权益替换，以计入平仓手续费且不虚构额外时间间隔。年化收益要求最终权益为正；波动率和 Sharpe 要求至少两个有效 bar 收益且分母有效，否则对应指标返回空值。总收益和最大回撤仍保留负权益结果，不把破产损失截断在 `-100%`。

## 7. 逐 bar 事件时序

`advance_bar(state, bar[T])` 采用以下严格顺序：

1. **校验输入**：验证时间戳严格递增、OHLC 合法，并将状态切换为 `RUNNING`。
2. **开盘盯市**：使用 `open[T]` 更新上一根结束时仍持有的仓位。
3. **跳空风险检查**：如果开盘已越过强平条件，先强平，不能再执行普通目标。
4. **资金费结算**：若本 bar 包含资金费率，对开盘前已经存在的仓位结算资金费，然后再次检查风险。
5. **执行待处理目标**：执行 `state.pending_target`，参考价和标记价均为 `open[T]`；执行后清空待处理目标。
6. **盘中强平检查**：多仓使用 `low[T]`、空头仓位使用 `high[T]` 作为不利价格检查强平。
7. **收盘盯市**：未强平时使用 `close[T]` 更新持仓和账户。
8. **完成持仓计时**：若本 bar 收盘仍有仓位，调用执行层的 bar 完成行为，使 `holding_steps` 增加一次。
9. **记录快照**：将本 bar 的输入、已执行目标、全部事件及收盘快照写入三个账本。
10. **提交游标**：更新 `current_index` 和 `last_bar`。如果已经强平，则状态改为 `LIQUIDATED` 并终止后续 bar。

`advance_bar` 返回后，上游或 `run_backtest` 才能提交 `generated_at = T` 的目标。`submit_target` 只把它保存为 `state.pending_target`，等待 `T+1`，不在本根成交。这样，动态仓位换算可以使用真实的 `close[T]` 账户权益，同时不会引入未来函数。

上述任一风险检查一旦触发强平，必须立即通过 `ExecutionEngine` 完成强平，并跳过剩余的资金费、普通目标和盯市步骤，只完成事件记录与游标提交；不得再排队新目标。

最后一根 K 线结束后：

- 默认以目标数量 `0`、原因 `END_OF_TEST` 按最后可用收盘价加平仓滑点成交并收取手续费；这是普通期末平仓，不设置 `is_liquidated`；
- 若配置为不强制平仓，则结果必须同时报告期末未实现盈亏和未平仓数量；
- 最后一根收盘生成的 `pending_target` 直接丢弃并记录为未执行目标，不得以最后收盘价成交；
- 不允许用数据区间之外的价格结算。

`finalize_backtest` 在没有任何 bar 时拒绝完成；状态已经是 `LIQUIDATED` 时只冻结结果，不重复平仓；正常结束时完成上述期末处理并把状态从 `RUNNING` 改为 `COMPLETED`。`COMPLETED` 或 `LIQUIDATED` 状态不能再次推进 bar 或提交目标；重复 finalize 不再修改状态，并返回等值结果。

该顺序明确了资金费、普通成交和强平在同一时间点发生时的优先级。`run_backtest` 只是组合调用这些编排函数，因此它必须与“逐根 `advance_bar` + `submit_target` + `finalize_backtest`”产生完全相同的结果。

## 8. 成交与持仓状态转换

令：

- `q0` 为执行前有符号持仓数量；
- `qt` 为目标有符号持仓数量；
- `dq = qt - q0` 为有符号成交数量；
- `dq > 0` 是买入，`dq < 0` 是卖出。

当 `abs(dq) <= quantity_epsilon` 时不产生成交和手续费，但仍执行盯市。

| 执行前 | 成交 | 目标结果 | 处理 |
| --- | --- | --- | --- |
| 空仓 | 买入 | 多仓 | 开多 |
| 空仓 | 卖出 | 空头 | 开空 |
| 多仓 | 买入 | 更大多仓 | 加多 |
| 多仓 | 卖出且不足现有数量 | 较小多仓 | 减多，结算对应盈亏 |
| 多仓 | 卖出且等于现有数量 | 空仓 | 平多 |
| 多仓 | 卖出且超过现有数量 | 空头 | 先平多，再以剩余数量开空 |
| 空头 | 卖出 | 更大空头 | 加空 |
| 空头 | 买入且不足现有绝对数量 | 较小空头 | 减空，结算对应盈亏 |
| 空头 | 买入且等于现有绝对数量 | 空仓 | 平空 |
| 空头 | 买入且超过现有绝对数量 | 多仓 | 先平空，再以剩余数量开多 |

反手在账务上拆成“关闭旧仓”和“打开新仓”两段，但对外可以是一笔成交：

- 全部 `abs(dq)` 按同一个滑点后成交价收费；
- 只有关闭旧仓的数量产生已实现盈亏；
- 新方向剩余数量的开仓均价等于本次成交价；
- `holding_steps` 重置为零。

## 9. 统一计算公式

以下公式均适用于 USDT 本位线性合约。

其中 `contract_size`、`leverage` 和 `maintenance_margin_rate` 均读取自当前 `Position.config`。

### 9.1 成交价格与名义价值

买入成交价：

`fill_price = reference_price × (1 + slippage_rate)`

卖出成交价：

`fill_price = reference_price × (1 - slippage_rate)`

成交名义价值：

`trade_notional = abs(dq) × fill_price × contract_size`

手续费：

`fee = trade_notional × fee_rate`

`slippage_rate` 必须是比例而不是绝对价格。零滑点时成交价等于参考价。

### 9.2 同方向加仓均价

仅在旧仓和成交同方向时计算：

`new_entry = (abs(q0) × old_entry + abs(dq) × fill_price) / abs(q0 + dq)`

减仓时剩余仓位的开仓均价不变；全平时归零；反手后新方向均价为本次成交价。

### 9.3 已实现盈亏

平仓数量：

`close_qty = min(abs(q0), abs(dq))`，仅当 `q0 × dq < 0` 时非零。

统一已实现盈亏：

`realized_pnl = close_qty × sign(q0) × (fill_price - entry_price) × contract_size`

手续费不混入该字段，而是单独从余额扣除。这样可以分别统计策略毛交易盈亏和交易成本。

### 9.4 未实现盈亏

`unrealized_pnl = position × (mark_price - entry_price) × contract_size`

因为 `position` 有符号，该公式同时适用于多仓和空头仓位。

### 9.5 持仓估值与保证金

以下字段由 `Position.revalue` 根据 `PositionConfig` 统一计算：

当前持仓名义价值：

`position_notional = abs(position) × mark_price × contract_size`

初始保证金占用：

`initial_margin = position_notional / leverage`

维持保证金：

`maintenance_margin = position_notional × maintenance_margin_rate`

维持保证金率必须作用于**持仓名义价值**，不能作用于初始保证金。`Position.revalue` 完成上述计算后，`ExecutionEngine` 将 `unrealized_pnl`、`initial_margin` 和 `maintenance_margin` 原样汇总到 `Account`。当前 `Account.update` 中的 `used_margin × maintenance_margin_rate` 会额外除以一次杠杆，必须删除，不能迁移为另一套账户公式。

### 9.6 账户恒等式

`balance = initial_balance + cumulative_realized_pnl - total_fee - total_funding`

其中 `total_funding` 为净支付金额，收到资金费时可以为负。

`equity = balance + unrealized_pnl`

`available_balance = equity - used_margin`

风险比率采用“维持保证金占权益比例”的口径：

```text
if maintenance_margin == 0:
    margin_ratio = 0
elif equity <= 0:
    margin_ratio = +inf
else:
    margin_ratio = maintenance_margin / equity
```

风险越高，`margin_ratio` 越大；权益为正时，`margin_ratio >= 1` 表示进入强平区。无持仓时比率为 `0`。强平的权威判断仍使用 `maintenance_margin > 0 and equity <= maintenance_margin`，不能只依赖除法结果；这样空仓但余额非正时不会被误标记为持仓强平。

当前代码使用 `equity / used_margin`，既没有使用维持保证金，方向也与目标风险比率相反，必须替换。`is_liquidated` 一旦触发仍保持为真，后续仓位归零不能把它恢复为假。

### 9.7 资金费

给定资金费率 `funding_rate`：

`funding_payment = position × mark_price × contract_size × funding_rate`

- 正资金费率时，多仓支付、空头仓位收取；
- `Account.pay_funding` 按 `balance -= funding_payment` 结算；
- 结算后立即重新计算权益并检查强平。

## 10. 原子执行流程

一次普通目标仓位执行应遵循以下流程：

1. 校验目标数量、参考价格、标记价格、配置和账户状态。
2. 使用当前标记价格刷新旧持仓与账户，保证决策基于最新状态。
3. 计算 `dq`；若为零，只返回盯市结果。
4. 根据买卖方向计算滑点后成交价。
5. 将交易分类为风险减少部分和风险增加部分。
6. 在临时状态上计算成交后的数量、均价、已实现盈亏、手续费、权益和保证金。
7. 对风险增加部分执行保证金校验。
8. 校验通过后一次性提交 `Position` 和 `Account`；失败时不得留下半更新状态。
9. 使用指定标记价格再次盯市，并检查是否因费用或滑点触发风险条件。
10. 生成执行结果与快照。

所有计算应先在局部临时值上完成，再提交真实对象。不能先调用 `pay_fee`，随后因保证金不足抛错，否则账户会在失败交易后错误损失手续费。

### 10.1 保证金不足处理

规则需要确定且可审计：

| 请求类型 | 保证金不足时的行为 |
| --- | --- |
| 纯减仓或平仓 | 始终允许执行 |
| 同方向加仓 | 整个增量拒绝，保持原仓位 |
| 从空仓开仓 | 整个请求拒绝，保持空仓 |
| 反手 | 先平掉旧仓；若新方向余额不足，则拒绝剩余开仓并停在空仓 |

反手被降级为平仓时，执行结果必须标记“目标未完全完成”，并记录实际成交数量及原因。任何自动缩小仓位的行为都必须显式记录，不能静默改变目标。

风险增加后的最低要求：

- `available_balance >= 0`；
- `equity > maintenance_margin`；
- 实际名义杠杆不超过允许值；
- 费用已经计入上述投影。

## 11. 强平模型

### 11.1 触发条件

每次盯市、成交和资金费结算后检查：

`maintenance_margin > 0 and equity <= maintenance_margin`

账户提交风险状态时使用粘性更新：`is_liquidated = is_liquidated or trigger`，不能在强平仓位归零后因 `maintenance_margin == 0` 自动恢复。

bar 内风险检查使用最不利价格：

- 多仓使用该 bar 的 `low`；
- 空头仓位使用该 bar 的 `high`。

如果最不利价格触发条件，第一版采用保守且确定的规则：以该最不利价格作为强平参考价，再叠加强平方向的滑点和可选强平费，将仓位全部关闭。

OHLC 无法还原 bar 内真实价格路径，因此该结果是保守近似。报告中必须注明 bar 级强平模型，不能将其表述为交易所逐笔级精度。

### 11.2 强平后的状态

- 仓位数量、开仓价、未实现盈亏和保证金归零；
- 平仓盈亏、普通手续费和强平费进入余额与累计项；
- `is_liquidated = True`，即使强平后因仓位归零而维持保证金变为零，也不能自动清除；
- 默认终止本次回测，不再接受策略目标；
- 余额允许保留负值用于审计，报告展示最终破产损失，不静默截断为零。

## 12. 示例：反手与后续盯市

配置：

- `contract_size = 1`；
- `leverage = 10`；
- `fee_rate = 0.0005`；
- `slippage_rate = 0`；
- `maintenance_margin_rate = 0.005`；
- 初始余额 `10,000 USDT`。

账户已有多仓 `q0 = +2`，开仓均价 `100`。策略在参考价 `110` 请求目标仓位 `qt = -1`：

1. `dq = -1 - 2 = -3`，即卖出 3 张；
2. 其中 2 张平多，1 张开空；
3. 已实现盈亏为 `2 × (110 - 100) = +20`；
4. 手续费为 `3 × 110 × 0.0005 = 0.165`；
5. 新仓位为 `-1`，新开仓均价为 `110`；
6. 新余额为 `10,000 + 20 - 0.165 = 10,019.835`；
7. 成交后以 `110` 盯市，未实现盈亏为零，初始保证金为 `110 / 10 = 11`；
8. 维持保证金为 `110 × 0.005 = 0.55`；
9. 账户风险比率为 `0.55 / 10,019.835 ≈ 0.0000549`，远小于强平阈值 `1`。

随后标记价跌至 `105`：

- 未实现盈亏为 `-1 × (105 - 110) = +5`；
- 权益为 `10,019.835 + 5 = 10,024.835`；
- 初始保证金为 `105 / 10 = 10.5`；
- 维持保证金为 `105 × 0.005 = 0.525`；
- 账户风险比率为 `0.525 / 10,024.835 ≈ 0.0000524`。

该示例同时验证反手拆分、空头盈亏符号、费用独立统计和维持保证金口径。

## 13. 必须始终成立的不变量

每次事件提交后至少检查：

1. `equity == balance + unrealized_pnl`；
2. `balance == initial_balance + realized_pnl - total_fee - total_funding`；
3. `Account.unrealized_pnl == Position.unrealized_pnl`；
4. `Account.used_margin == Position.initial_margin`；
5. `Account.maintenance_margin == Position.maintenance_margin`；
6. `Position.notional == abs(Position.position) × mark_price × contract_size`；
7. 无维持保证金时 `margin_ratio == 0`；有维持保证金且权益非正时为正无穷；其余情况为 `maintenance_margin / equity`；
8. 手续费、名义价值、初始保证金、维持保证金和 `margin_ratio` 均非负；
9. 空仓时开仓价、名义价值、浮盈、两类保证金和持仓时长为零；
10. 非空仓时开仓价和标记价格均为有限正数；
11. 减仓不会改变剩余持仓的开仓均价；
12. 同方向加仓不会产生已实现盈亏；
13. 目标等于当前仓位时不产生手续费；
14. 回测时间戳严格递增，信号执行时间晚于信号生成时间；
15. `is_liquidated` 一旦为真，在 reset 前不能恢复为假；
16. `BacktestState` 中不存在 `balance`、`equity`、`position_quantity` 等标量形式的重复财务字段；
17. `pending_target` 非空时，其 `generated_at` 必须等于 `last_bar.timestamp`；
18. 账本只追加且记录的是事件提交后的不可变快照；
19. `run_backtest` 与逐根调用 `advance_bar`、`submit_target` 后再调用 `finalize_backtest` 的结果一致。

浮点比较使用统一容差，不使用裸 `==` 判断金额或数量。

## 14. 异常和边界规则

- 非有限数量或价格、非正价格：拒绝执行并报参数错误；
- 负手续费率、负滑点率、负维持保证金率、非正杠杆或合约面值：构造配置时立即报错；
- 目标与当前仓位之差小于数量容差：视为无成交；
- 手续费或资金费使账户进入强平区：完成结算后立即进入强平流程；
- 资金费为负：允许，表示账户收到资金费；
- 已强平账户：只允许读取快照和 reset，不允许普通交易；
- 数据时间倒序或重复：回测开始前拒绝；
- OHLC 非法，例如 `low > high` 或价格非正：数据校验阶段拒绝；
- 目标时间戳找不到对应 bar、重复目标或目标数量非有限：输入校验阶段拒绝；
- 最后一根 bar 没有下一根开盘：不得执行新信号，只执行配置允许的期末平仓；
- 结果中的拒绝不是异常中断，而是带原因的业务结果；输入和内部不变量错误才抛异常。

## 15. 与现有代码的迁移关系

### 15.1 明确废弃的能力

- `engine.py` 中的 `BacktestEngine` 整体废弃，不从中抽取事件循环、收益公式或状态管理逻辑；
- `_attach_trading_results` 的向量化仓位、手续费、收益和权益计算整体废弃；
- 旧 `BacktestResult` 与 fast/slow、pandas DataFrame 耦合，不能作为新的核心结果类型；
- `backtest_handler.py` 对旧 `BacktestEngine` 的直接构造需要改成输入适配、调用 `run_backtest` 和报告输出三步；
- 旧 HTML 报告只有在改为消费新 `BacktestResult` 后才能保留，不作为核心系统依赖。

废弃代码不得用于兼容结果对照，不参与新实现的设计决策、导入关系或测试夹具；新系统以本设计中的公式、不变量和手工可验证场景作为唯一正确性基线。

### 15.2 保留的领域语义

- 保留“bar 收盘生成目标、下一根开盘执行”的无未来函数约定；
- 保留 `Account` 中余额、权益、费用和强平相关字段；
- 保留 `Position` 的有符号持仓表达；
- 保留目标持仓持续到下一次目标变化的含义；
- 模型推理和 fast/slow 信号可以继续存在，但只能作为 `TargetInstruction` 的上游生产者。

### 15.3 新实现所需调整

- 将 `Position` 改为实例级 dataclass，并提供明确 reset/snapshot 行为；
- 新建 `PositionConfig`，将 `leverage`、`contract_size` 和 `maintenance_margin_rate` 从账户或执行配置迁入持仓配置；
- 将现有 `Position.margin` 明确重命名为 `initial_margin`，新增 `notional` 和 `maintenance_margin`；
- 删除 `Account.update` 内部的维持保证金公式，改为接收 `Position` 的估值汇总；
- 将 `margin_ratio` 改为 `maintenance_margin / equity` 风险比率，并修正空仓、非正权益和强平粘性边界；
- 将 `slippage` 明确命名为比例 `slippage_rate`；
- 将 `ExecutionEngine` 移到独立的 `execution.py`，且不导入旧 `engine.py`；
- 新建 `simulation.py`，以 `BacktestState` 和模块级函数组织逐 bar 编排；
- 新建不依赖 pandas 的 `MarketBar`、`TargetInstruction`、快照和账本值对象；
- 在策略适配层完成信号或目标杠杆到目标合约数量的换算；
- 从真实权益快照派生每 bar 收益以及收益率、波动率和回撤指标；
- 统一 backtest 与 RL 环境的交易成本、目标杠杆和时序定义，避免两套公式长期漂移。

### 15.4 新系统行为基线

建立不依赖任何旧输出的手工可验证场景：

- 使用少量固定 OHLC bar 和明确的目标合约数量；
- 滑点、资金费和强平费先设为零，再逐项开启验证；
- 覆盖 `0 -> +1 -> +2 -> 0 -> -1 -> +1` 的完整转换；
- 每个目标严格在生成后的下一根开盘执行；
- 逐笔手算成交价、余额、权益、保证金和最终指标，并与三个账本对账。

这组确定性 fixture 与第 9 节公式、第 13 节不变量共同构成迁移验收真值。

## 16. 测试策略

### 16.1 公式单元测试

- 多仓和空头仓位的未实现盈亏；
- 同方向加仓均价；
- 多空减仓已实现盈亏；
- 全平与双向反手；
- 买卖滑点方向；
- 手续费、初始保证金、维持保证金和资金费；
- 同一名义价值改变杠杆时，初始保证金变化但维持保证金不变；
- `margin_ratio` 在空仓、正常风险、恰好达到 `1`、权益非正时的边界；
- 空仓且余额非正不会新触发持仓强平，已触发的 `is_liquidated` 在平仓后保持为真。

### 16.2 状态转换表测试

覆盖第 8 节全部状态组合，并在每一步验证第 13 节的不变量。特别测试：

- `+2 -> -1` 和 `-2 -> +1`；
- 盈利减仓与亏损减仓；
- 目标不变；
- 反手只完成平仓部分；
- 费用使可用余额不足；
- 资金费触发强平。

### 16.3 属性测试

生成随机但有效的价格与目标仓位序列，验证：

- 任意序列下账户恒等式成立；
- 零成本下开仓后按原价平仓，最终交易盈亏为零；
- 同一成交拆成多次、且成交价相同时，最终数量和均价一致；
- 将所有持仓和价格方向同时取反时，多空公式保持对称；
- 同一输入重复运行结果完全一致。

### 16.4 集成测试

- 只使用内存中的 `MarketBar` 和 `TargetInstruction`，不加载模型、数据库或 DataFrame，也能完成回测；
- 信号 `T` 只能影响 `T+1` 开盘后的仓位；
- 开盘跳空强平优先于普通目标成交；
- bar 内不利价格能触发强平；
- 期末强制平仓只使用最后一个可用价格；
- 最后一根 bar 生成的目标不会成交；
- `run_backtest` 与逐根调用 `advance_bar`、`submit_target` 后再调用 `finalize_backtest` 产生相同状态、账本和指标；
- bar、trade、account 三个账本可以互相对账；
- 总收益、年化收益、波动率、Sharpe、回撤、交易次数、手续费和持仓时间均可从账本重算；
- 核心模块的导入图中不存在对旧 `engine.py`、模型、训练器、pandas、PyTorch 或报告模块的依赖。

## 17. 分阶段实施顺序

### 阶段一：数据模型与纯计算

- 固化单位、配置校验和状态不变量；
- 完成 `PositionConfig`、`Position`、`Account` 的实例状态与快照；
- 完成 `Position.revalue`、账户聚合、费用、滑点、均价、盈亏和风险比率的纯计算测试。

### 阶段二：成交状态机

- 完成买卖路由及所有仓位转换；
- 完成风险增加校验、拒绝结果和原子提交；
- 完成执行结果与 trade ledger。

### 阶段三：独立回测编排

- 完成 `MarketBar`、`TargetInstruction`、`BacktestState` 和新 `BacktestResult`；
- 完成 `advance_bar`、`submit_target`、`finalize_backtest`、`run_backtest`、资金费、盘中强平和期末处理；
- 完成 bar/account ledger，以及只从权益账本计算的收益、波动率和回撤指标；
- 用内存行情和手写目标序列证明三大核心组件可以独立完成回测。

### 阶段四：接入上游与展示层

- 将 fast/slow 信号通过适配器转换为 `TargetInstruction`；
- 改造 handler 和 HTML 报告以消费新的 `BacktestResult`；
- 统一 RL 环境与回测执行语义。

### 阶段五：删除废弃实现

- 删除 `BacktestEngine`、旧 `BacktestResult` 和旧向量化账务辅助函数；
- 清理所有旧导入与兼容分支；
- 再次运行核心独立性、账务不变量和报告集成测试。

## 18. 验收标准

设计实现完成时应同时满足：

- 状态转换表全部有测试且通过；
- 账户与持仓不变量在每个事件后成立；
- 不存在当根收盘信号在当根开盘成交的未来函数；
- 所有余额变化都能定位到成交、资金费或强平事件；
- 保证金不足不会留下部分写入的错误状态；
- 强平路径可以由 OHLC 和配置确定性复现；
- 回测指标可以只依靠结果账本重新计算；
- `ExecutionEngine` 不依赖模型、pandas DataFrame 或报告模块；
- `Account` 不持有杠杆、合约面值或维持保证金率，也不自行计算持仓保证金；
- `Position` 是名义价值、初始保证金和维持保证金的唯一计算来源，账户汇总值可与其逐项对账；
- `margin_ratio` 随风险上升而增大，空仓为 `0`，达到 `1` 时与直接强平条件一致；
- `BacktestState` 只组织 `Account`、`Position`、`ExecutionEngine`、时间状态和账本，不复制财务事实；
- 只传入标准 bar 和目标合约数量即可运行完整回测；
- 批量入口与“逐 bar + submit + finalize”入口产生完全相同的结果；
- 新核心代码对废弃 `BacktestEngine` 和旧 `BacktestResult` 的依赖数为零；
- 同一目标仓位序列可被普通策略和 RL 策略共同复用。

## 19. 结论

目标系统由 `Position`、`Account` 和 `ExecutionEngine` 构成交易模拟内核，由 `BacktestState` 聚合单次运行所需对象、待执行目标和账本，再由 `advance_bar`、`submit_target`、`finalize_backtest` 与 `run_backtest` 负责确定性的时间编排。给定标准化行情和目标合约数量，这套系统自身即可完成成交、盯市、资金费、强平、期末结算、审计记录和指标计算。

现有 `BacktestEngine` 不属于目标架构。上游模型和策略只生产标准目标，下游报告只消费冻结结果；两端都不能进入核心账务与编排路径。
