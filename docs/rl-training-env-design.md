# CoinQuant RL 交易训练系统设计

> 状态：设计提案
> 日期：2026-08-05
> 范围：`src/coinquant/rl` 下的观测、动作、奖励、Gymnasium 环境和 PPO 训练编排

## 1. 目标与边界

RL agent 的输入由三部分组成：

1. 最近 `window_size`（默认 32）根已完成 K 线的基础行情特征；
2. 现有 fast、slow 两个 DL 模型在这些 K 线上的冻结预测特征；
3. 当前账户和持仓状态。

Agent 在 bar `T` 收盘后产生一个目标暴露，环境在 bar `T+1` 开盘执行，随后使用 `T+1` 的行情完成盯市并计算奖励。这样训练环境与独立回测系统保持相同的时序语义：决策不读取未来的开盘价、最高价、最低价或收盘价。

第一版只支持：

- 单账户、单品种、USDT 本位线性合约；
- 单向净持仓、全仓保证金；
- 连续动作、按 `rebalance_interval` 周期调整目标仓位，间隔内保持上一目标；
- 固定手续费、滑点、账户杠杆和维持保证金率；
- fast/slow DL 模型冻结，只做推理，不在 RL 训练中更新；
- PPO 训练，按时间顺序使用 train/valid/test 三个数据段；训练器也支持实验性的 SAC。

第一版不支持：

- 多品种共享保证金、逐仓和组合保证金；
- 限价单、部分成交、成交量约束和盘口撮合；
- RL 与 DL 联合端到端训练；
- 在线改变账户杠杆；
- 使用 future return、label 或其他未来字段作为观测。

## 2. 系统边界与组件关系

```text
原始 OHLCV
    |
    v
基础特征构造 -----> fast DL 推理 ----┐
                     slow DL 推理 ----┼--> RLFeatureFrame
                                      |
                                      v
                              TradingEnv (Gymnasium)
                              /       |          \\
                         observation action      reward
                              \\       |          /
                                      v
                           Account / Position / ExecutionEngine
                                      |
                                      v
                             PPO Trainer / Evaluator
```

RL 环境可以复用 `backtest/account.py`、`backtest/position.py` 和 `backtest/execution.py` 的交易账务语义，但不能导入或调用废弃的 `backtest.engine.BacktestEngine`。`TradingEnv` 负责时间推进和 Gymnasium 接口，交易公式仍由执行核心负责。

## 3. 数据契约与时间语义

### 3.1 行情数据

环境输入为按 `open_time` 严格递增的内存表格，每行至少包含：

| 字段 | 含义 |
| --- | --- |
| `open_time` | bar 开始时间，必须唯一且递增 |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `funding_rate` | 可选资金费率，没有时按 0 处理 |

价格必须为有限正数，并满足：

```text
low <= min(open, close) <= max(open, close) <= high
```

环境在构造时复制并校验数据，不修改调用方的 DataFrame，也不依赖数据库、pandas 之外的外部状态。

### 3.2 基础行情特征

每根 K 线生成以下 5 个基础特征，均为无量纲或对数变化，避免不同品种价格尺度影响策略：

```text
open_return       = open_t / close_(t-1) - 1
high_return       = high_t / open_t - 1
low_return        = low_t / open_t - 1
close_return      = close_t / open_t - 1
log_volume_change = log(1 + volume_t) - log(1 + volume_(t-1))
```

第一根记录无法计算 `open_return` 和 `log_volume_change` 时，向前填充为 0；所有特征必须经过有限值检查。若未来要加入 funding、波动率或技术指标，必须增加版本化的特征契约，不能隐式改变观测维度。

### 3.3 fast/slow DL 特征

两个模型分别产生一个标量：

- `prediction_fast`；
- `prediction_slow`。

推理流程：

1. 从模型元数据读取 `feature_columns`、模型类型、模型参数、品种和周期；
2. 使用与 DL 训练完全相同的特征构造逻辑；
3. 对每个时间 `T`，只使用截至 `T` 的序列生成预测；
4. 两个模型均切换到 eval 模式并关闭梯度；
5. 按 `open_time` 内连接 fast/slow 预测，缺失任一预测的行从可训练起点之前剔除；
6. 将预测写入内存 `RLFeatureFrame`，环境 step 不重复加载模型或执行推理。

DL 模型可以使用已有的 128 根序列输入，因此 RL 的最早可决策位置是：

```text
max(window_size, fast_sequence_length, slow_sequence_length) - 1
```

模型标签字段（例如 `label_z_score_close_fast`、`label_z_score_close_slow`）、`close_fast`、`close_slow`、未来收益和任何由未来 K 线计算的字段只能用于 DL 训练或离线评估，禁止进入 RL observation。

### 3.4 数据切分

数据按时间切分为 train、valid、test，不允许随机打乱后跨段采样：

- train：训练 PPO 和拟合观测归一化统计量；
- valid：确定模型选择、早停和超参数，不更新归一化统计量；
- test：只在最终模型确定后运行一次。

fast/slow 模型的 checkpoint 必须在 RL 训练开始前固定。RL 训练过程中不得根据 valid/test 结果重新训练或选择 DL 模型。

## 4. 文件职责与公开对象

| 文件 | 责任 | 主要对象 |
| --- | --- | --- |
| `rl/observation.py` | 特征列、DL 预测适配、观测窗口和归一化 | `ObservationConfig`、`ObservationBuilder`、`DLFeatureProvider` |
| `rl/action.py` | 连续动作定义、裁剪、目标暴露转换 | `ActionConfig`、`ActionAdapter` |
| `rl/reward.py` | 收益、风险和回撤奖励 | `RewardConfig`、`RewardBreakdown`、`RewardCalculator` |
| `rl/env.py` | Gymnasium reset/step、时间游标、交易核心协调 | `TradingEnv`、`EnvConfig` |
| `rl/trainer.py` | 数据准备、PPO/SAC、验证、保存和评估 | `RLTrainingConfig`、`RLTrainingArtifacts`、`train_rl` |
| `rl/features.py` | 可选的 Conv1d+GRU 时序特征提取 | `TemporalFeatureExtractor` |
| `rl/ensemble.py` | 多随机种子策略的推理集成 | `RLPolicyEnsemble`、`load_ensemble` |
| `rl/report.py` | 将 RL ledger 与 OHLCV 对齐并生成成交标注页面 | `RLTradeReport`、`render_rl_report` |

这些模块不应该导入旧的 `BacktestEngine`。报告和指标只消费环境产生的冻结 episode 记录。

## 5. Observation 设计

### 5.1 观测结构

每个决策时刻 `T` 的行情窗口为：

```text
[T - 9, T - 8, ..., T]
```

每根 K 线有 7 个市场通道：

```text
[open_return,
 high_return,
 low_return,
 close_return,
 log_volume_change,
 prediction_fast,
 prediction_slow]
```

此外拼接 4 个账户状态通道：

```text
[current_exposure,
 equity_ratio,
 drawdown,
 rolling_volatility]
```

因此默认观测为一个 `float32` 向量，长度为：

```text
window_size * 7 + 4 = 10 * 7 + 4 = 74
```

上式是本设计最初的 10-bar/标量预测基线。当前代码会在存在 `feat_*`、预测向量和预测上下文时动态扩展市场通道，并将账户状态按历史长度展开；实际维度以 `ObservationBuilder.observation_size` 为准。

第一版固定返回扁平向量，以便直接接入 Stable-Baselines3 的 `MlpPolicy`。训练器可选 `TemporalFeatureExtractor` 将该扁平向量还原为市场/账户时间序列，经 Conv1d+GRU 编码后再交给 actor/critic；关闭该选项则使用普通 MLP。

### 5.2 账户状态定义

```text
current_exposure = current_position_notional / equity
equity_ratio     = equity / initial_equity
drawdown         = max(0, 1 - equity / peak_equity)
rolling_volatility = 最近 risk_window 个 bar 收益的样本标准差
```

当权益小于等于 0 时，所有账户状态使用有限的边界值，并结束 episode。账户状态不能使用未来 bar 的数据。

### 5.3 归一化与观测空间

环境声明：

```python
spaces.Box(low=-clip_value, high=clip_value, shape=(74,), dtype=np.float32)
```

归一化规则：

- 基础收益和 DL 输出先检查有限性，再按训练集统计量标准化；
- `equity_ratio` 以初始权益为基准；
- `drawdown` 限制在 `[0, 1]`；
- `current_exposure` 限制在 `[-max_leverage, max_leverage]`；
- 所有通道最终裁剪到 `[-clip_value, clip_value]`，默认 `clip_value=10`；
- valid/test 只读取 train 拟合的均值、标准差和裁剪参数。

禁止在每个 episode 或每个 bar 上重新拟合归一化统计量，否则会引入隐含未来信息并导致训练、验证和线上分布不一致。

## 6. Action 设计

### 6.1 动作空间

第一版使用一个连续标量：

```python
action_space = spaces.Box(
    low=np.array([-max_leverage], dtype=np.float32),
    high=np.array([max_leverage], dtype=np.float32),
    dtype=np.float32,
)
```

动作 `a_T` 的语义是 bar `T` 收盘后希望在下一根 bar 形成的目标暴露，而不是买卖数量增量：

- `a_T > 0`：目标多仓；
- `a_T < 0`：目标空仓；
- `a_T = 0`：目标空仓；
- `|a_T|`：目标名义价值相对权益的倍数。

策略动作必须先裁剪到合法范围，再转换为目标合约数量。动作不携带杠杆，账户杠杆由 `AccountConfig` 固定提供。

### 6.2 暴露到合约数量的转换

在 `T+1` 开盘真正执行时，使用当时可用的权益和开盘价计算目标数量：

```text
target_notional = clipped_action × equity_at_execution
target_quantity = target_notional
                  / (open_(T+1) × contract_size)
```

该转换发生在成交时，不能在 `T` 收盘使用未知的 `T+1` 开盘价提前计算。`target_quantity` 随后交给 `ExecutionEngine.execute_target()`，由执行核心统一处理费用、滑点、保证金、反手和强平。

`max_leverage` 是 RL 目标暴露上限（目标名义价值 ÷ 当前权益），不是账户的保证金杠杆；账户/合约保证金杠杆由 `AccountConfig.leverage`（训练配置中的 `account_leverage`）决定。默认不得大于账户杠杆；如果动作转换后保证金不足，环境记录拒绝或反手降级，不静默改变 agent 的动作语义。

## 7. 环境时序与 Gymnasium API

### 7.1 `reset`

`reset(seed, options)`：

1. 重置账户、持仓、交易核心、时间游标、峰值权益和奖励历史；
2. 从第一个有完整 `window_size` 和 fast/slow 预测的 bar 开始；
3. 不执行任何交易；
4. 返回 bar `T` 收盘时的 observation 和 info。

`reset` 必须支持相同 seed 得到相同起点和相同数据顺序。默认不随机选择 episode 起点；如需随机起点，必须只在 train 环境启用，并记录起点时间。

### 7.2 `step`

给定 bar `T` 的 observation 和动作 `a_T`：

1. 校验并裁剪动作；
2. 将目标暴露放入 pending action；
3. 推进到 bar `T+1`；
4. 以 `T+1` 开盘价盯市并执行 pending target；
5. 对 funding、盘中不利价格和收盘价执行同一套风险处理；
6. 以 `T+1` 收盘权益与 `T` 收盘权益计算 reward；
7. 返回 `observation_(T+1), reward, terminated, truncated, info`。

当 `rebalance_interval > 1` 时，策略仍可每个 bar 输出 action，但只有调仓时点更新目标仓位并调用 `execute_target`；其余 bar 保持实际持仓，不会为了重新计算名义数量而产生重复成交。`min_rebalance_notional_ratio` 则在调仓时点进一步跳过小于权益比例阈值的名义变动；方向反转通常会自然超过该阈值。

环境 step 不允许读取 `T+1` 的 high/low/close 来决定 `T+1` 开盘订单数量，也不允许把 `T+1` 的模型预测放入 `observation_T`。

### 7.3 结束条件

`terminated=True` 的情况：

- 到达数据集最后一个可执行 bar；
- 账户权益小于等于 0；
- 执行核心触发强平；
- 显式触发风险终止条件。

`truncated=True` 只用于外部的最大 episode 步数限制，不与交易风险终止混用。

默认在最后一个可执行 bar 的收盘价平掉剩余仓位，最后一步的费用和已实现盈亏进入最终权益。环境返回的 info 标记 `end_of_episode_close=True`。

### 7.4 `info` 最低字段

每个 step 的 `info` 至少包含：

```text
decision_time
entry_time
exit_time
action
target_exposure
previous_exposure
actual_position
gross_return
net_return
turnover
fee_cost
slippage_cost
funding_payment
base_reward
risk_penalty
drawdown_penalty
reward
equity
peak_equity
drawdown
liquidated
execution_event_type
```

这些字段用于训练后对账和诊断，不作为 observation 的隐藏输入。

## 8. Reward 设计

### 8.1 收益基准

账户权益变化是 reward 的权威收益来源：

```text
net_return_T = equity_(T+1) / equity_T - 1
```

当 `equity_T <= 0` 时该比率不再计算，环境将 episode 标记为终止，并使用配置的终止奖励规则记录最后一次权益变化。

手续费、滑点和资金费已经通过 `Account` 进入权益，不能在 reward 中再次扣除。为方便对比，同时记录：

```text
gross_return_T = position_exposure_T × market_return_(T+1)
```

默认基础奖励使用简单收益率，也支持对权益为正时使用对数收益：

```text
base_reward = net_return_T
或
base_reward = log(1 + net_return_T)
```

### 8.2 风险惩罚

滚动波动率使用截至当前 bar 已知的净收益：

```text
rolling_volatility_T = std(net_return_{T-risk_window+1:T}, ddof=1)
```

风险惩罚由暴露相关波动风险和仓位复杂度组成：

```text
risk_penalty = volatility_penalty
                × abs(target_exposure_T)
                × rolling_volatility_T
              + position_penalty
                × target_exposure_T²
turnover_penalty = turnover_penalty_rate × turnover_T
short_penalty = short_penalty_rate × newly_opened_short_exposure_T
```

样本数不足两个时滚动波动率按 0 处理。`position_penalty` 可设置为 0；它不是交易成本，不改变账户账本。

### 8.3 回撤惩罚

```text
peak_equity_T = max(initial_equity, equity_0, ..., equity_T)
drawdown_T = max(0, 1 - equity_T / peak_equity_T)
drawdown_penalty = drawdown_penalty_rate × drawdown_T
```

默认惩罚当前回撤水平，使 agent 避免长期处于深度回撤；若需要只惩罚回撤扩大，可以将输入替换为 `max(0, drawdown_T - drawdown_(T-1))`，但必须在配置中明确记录。

### 8.4 总奖励

```text
raw_reward = base_reward - risk_penalty - drawdown_penalty
             - turnover_penalty - short_penalty - liquidation_penalty
reward = reward_scale × raw_reward
```

上述扣减只在 `enable_penalties=True` 时生效；默认关闭时 `raw_reward=base_reward`。

强平或权益归零时，正常 reward 仍按最后一次权益变化计算；如需额外惩罚，只能作为明确的 `liquidation_penalty` 配置，不能隐藏在交易成本中。

奖励组件必须全部写入 `info`，训练报告同时保存 `total_reward`、`mean_reward`、`total_return`、`max_drawdown` 和成本统计，避免只看 reward 判断策略质量。

## 9. 配置对象

### 9.1 `ObservationConfig`

```text
window_size: int = 32
basic_feature_columns: tuple[str, ...]
prediction_columns: tuple[str, str] = ("prediction_fast", "prediction_slow")
account_feature_columns: tuple[str, ...]
account_history_length: int | None = None
include_dl_features: bool = True
include_prediction_vectors: bool = True
include_prediction_context: bool = True
dl_feature_prefix: str = "feat_"
clip_value: float = 10.0
normalize: bool = True
```

### 9.2 `ActionConfig`

```text
max_leverage: float = 1.0
quantity_epsilon: float = 1e-12
```

### 9.3 `RewardConfig`

```text
reward_mode: str = "simple"       # simple 或 log
reward_scale: float = 100.0
drawdown_penalty_rate: float = 0.0
volatility_penalty: float = 0.0
position_penalty: float = 0.0
turnover_penalty_rate: float = 0.0
short_penalty_rate: float = 0.0
risk_window: int = 20
liquidation_penalty: float = 0.0
enable_penalties: bool = False
```

### 9.4 `EnvConfig`

```text
account_config: AccountConfig
execution_config: ExecutionConfig
initial_equity: float | None = None  # None 时使用 account_config.initial_balance
force_close_at_end: bool = True
max_episode_steps: int | None = None
rebalance_interval: int = 1
min_rebalance_notional_ratio: float = 0.0
```

`initial_equity` 只是 RL 归一化的显式别名；如果提供，必须等于 `account_config.initial_balance`，不能形成第二套账户余额来源。

### 9.5 `RLTrainingConfig`

```text
symbol: str
period: str
window_size: int = 32
account_history_length: int | None = None
include_dl_features: bool = True
include_prediction_vectors: bool = True
include_prediction_context: bool = True
dl_vector_dim: int = 8
max_leverage: float = 0.5
account_leverage: float = 2.0
rebalance_interval: int = 48
total_timesteps: int = 100_000
eval_freq: int = 20_000
seed: int = 59_483
learning_rate: float = 3e-4
n_steps: int = 2048
batch_size: int = 256
gamma: float = 0.999
gae_lambda: float = 0.95
ent_coef: float = 0.0
normalize: bool = True
device: str = "cpu"
output_dir: str | None = None
```

训练配置必须同时记录 fast/slow checkpoint 路径、DL 元数据路径、特征版本、数据切分时间、手续费、滑点、账户杠杆、环境配置和随机种子。

## 10. PPO 训练与评估

### 10.1 训练流程

```text
build train/valid/test raw frames
        |
        v
construct basic features and frozen fast/slow predictions
        |
        v
fit train observation normalizer
        |
        v
DummyVecEnv(TradingEnv) + optional VecNormalize
        |
        v
PPO.learn(total_timesteps)
        |
        +--> deterministic validation evaluation
        |
        v
freeze best model and normalizer
        |
        v
single deterministic test evaluation
```

训练环境可使用 `DummyVecEnv`；第一版不要求多环境并行，避免时间序列切分和随机状态管理复杂化。`VecNormalize` 的统计量只在 train 阶段更新，valid/test 使用 `training=False` 和 `norm_reward=False`。

验证回调必须：

- 使用独立 valid 环境；
- 不修改 train 环境状态；
- 使用 deterministic action；
- 至少记录最终权益、总收益、最大回撤、Sharpe、成交成本和强平次数；
- 按验证指标保存 best model，而不是只按训练 reward 保存。

### 10.2 训练产物

每次训练输出独立目录，例如：

```text
data/model/rl/ppo_<symbol>_<period>_<timestamp>/
├── config.json
├── model/
│   ├── final_model.zip
│   ├── best_model.zip
│   ├── final_vecnormalize.pkl
│   └── best_vecnormalize.pkl
├── metrics.json
├── train_ledger.jsonl
├── valid_ledger.jsonl
└── test_ledger.jsonl
```

`RLTrainingArtifacts` 至少包含：

```text
run_dir
final_model_path
best_model_path
final_vecnormalize_path
best_vecnormalize_path
metrics_path
train_metrics
valid_metrics
test_metrics
```

模型、归一化统计量、环境配置和 DL checkpoint 必须能够共同恢复一次完全相同的评估环境。

## 11. 评估指标与对账

训练报告至少包含：

- `total_return`、`annualized_return`；
- `annualized_volatility`、`sharpe`、`max_drawdown`；
- `final_equity`、`total_reward`、`mean_reward`；
- `trade_count`、`total_turnover`、`avg_turnover`；
- `total_fee`、`total_funding`、`liquidation_count`；
- `exposure`、多仓/空头/空仓比例；
- 训练、验证和测试的起止时间。

评估结果不能只根据 reward 排名。至少需要将 RL 的成交账本与以下恒等式对账：

```text
equity = balance + unrealized_pnl
balance = initial_equity + realized_pnl - total_fee - total_funding
```

同一输入、同一配置和同一 seed 下，重复评估应得到相同动作、权益曲线和交易账本（允许浮点容差）。

## 12. 测试要求

### 12.1 观测测试

- 窗口始终包含恰好最近 `window_size`（默认 32）根 bar；
- 第一根可决策 bar 不使用未完成窗口；
- fast/slow 预测与时间戳正确对齐；
- 修改未来 bar 不会改变当前 observation；
- 所有 observation 有限且形状固定；
- valid/test 使用 train 的归一化统计量。

### 12.2 动作测试

- `NaN`、`inf` 动作被拒绝；
- 动作超出范围时裁剪并记录；
- 正负动作分别产生多空目标；
- `a_T` 只能影响 `T+1` 开盘，不影响 `T` 收盘权益；
- 目标暴露转换后的合约数量与开盘价格一致；
- 非调仓 bar 保持上一目标仓位且不重复产生执行成交；
- 保证金不足时不静默改变动作语义。

### 12.3 奖励测试

- 无仓位、无费用时收益奖励为 0；
- 正收益奖励为正，负收益奖励为负；
- 手续费、滑点、资金费只计入一次；
- 波动率增加时风险惩罚不减小；
- 回撤增加时回撤惩罚不减小；
- reward 组件之和等于 `raw_reward × reward_scale`。

### 12.4 环境和训练测试

- `reset` 后第一步状态确定；
- `step` 后时间游标只前进一个 bar；
- 最后一根 bar 目标不产生未来成交；
- 强平后 episode 终止；
- train/valid/test 时间无重叠；
- 单步环境与批量评估的账本一致；
- PPO 模型、VecNormalize 和配置可以重新加载并复现测试结果。

## 13. 实现顺序

1. 实现 `ObservationConfig`、基础特征和 fast/slow 预测对齐；
2. 实现 `ActionConfig` 和目标暴露到合约数量的转换；
3. 实现 `RewardConfig`、奖励分解和历史统计；
4. 实现 `TradingEnv`，接入 `Account/Position/ExecutionEngine`；
5. 编写观测、动作、奖励和时序测试；
6. 实现 PPO trainer、验证回调和产物保存；
7. 用固定 seed 完成 train/valid/test 端到端评估；
8. 将 RL ledger 接入独立回测报告，不重新引入 `BacktestEngine`。

## 14. 当前实现补充（以代码为准）

本设计最初以 10 根 K 线和两个标量 DL 预测为基线；当前实现已经扩展为：

- `ObservationConfig`/`RLTrainingConfig` 默认使用 32 根行情窗口，账户状态也按同长度保存历史序列；
- 当输入帧包含 `feat_*` 列时，观测自动纳入完整的 DL 训练特征；
- 旧标量 checkpoint 仍兼容，同时从 Transformer 最后 GRU 状态提取可配置的 8 维向量，并生成预测变化、快慢价差、滚动均值/波动等因果上下文；
- `include_dl_features`、`include_prediction_vectors`、`include_prediction_context` 和 `dl_vector_dim` 可用于做消融对照；
- PPO 默认策略网络为 `256-256-128` MLP；
- `rebalance_interval` 控制目标仓位的最小调仓间隔；默认 48 根 bar，非调仓 bar 保持上一目标且不重复执行成交；默认 `max_leverage=0.5`，`0.25` 可作为防守模式；
- 默认 `account_leverage=2.0`（合约保证金杠杆），与目标暴露上限分别控制不同风险维度；
- `turnover_penalty_rate` 按实际成交换手惩罚，`short_penalty_rate` 按新增空头暴露惩罚；两者与风险、回撤和强平惩罚一样保留为显式实验参数，默认由 `enable_penalties=False` 关闭，基础 reward 主要使用账户净收益。

以上行为以 `src/coinquant/rl` 和 `src/coinquant/model/transformer.py` 的实现为准。
