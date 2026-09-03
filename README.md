# BTC 量化交易平台

## RL 训练

```bash
coinquant train-rl --symbol BTC/USDT --period 1h --timesteps 100000
```

CLI 只暴露品种、周期和训练总步数；PPO rollout、batch 和验证频率会根据总步数自动设置，其余交易和奖励参数使用 `RLTrainingConfig` 的统一默认值。

当前 RL 默认配置使用 32 根历史 K 线；观测会自动包含 DatasetBuilder 生成的 `feat_*` 特征、fast/slow 模型的标量预测及 8 维末层隐藏向量、预测变化/价差等因果上下文，并附加与行情窗口等长的账户状态历史。PPO 策略网络默认为 `256-256-128` 三层 MLP。默认目标仓位上限为 `0.5x`、账户/合约保证金杠杆为 `2x`，每 48 根 K 线才执行一次新目标仓位（中间保持持仓）；`0.25x` 可作为防守模式。`min_rebalance_notional_ratio` 可进一步按名义变动比例跳过小额调仓；`turnover_penalty_rate` 和 `short_penalty_rate` 是可独立试验的奖励项，默认均为 0 且不会改变现有基线。风险、回撤和强平惩罚默认关闭（`enable_penalties=False`），reward 以账户净收益为主。

一次完整训练会在 `data/model/rl/ppo_<symbol>_<period>_<timestamp>/` 下保存 `model/best_model.zip` 与对应的 `model/best_vecnormalize.pkl`。防守型候选产物为 [ppo_BTC_USDT_1h_20260902_074408](data/model/rl_production/ppo_BTC_USDT_1h_20260902_074408/)（0.25x：验证 +6.98%，测试 +12.84%，测试最大回撤 -5.40%）。若希望使用至少半仓，可参考 [ppo_BTC_USDT_1h_20260902_091552](data/model/rl_grid_x05_seed12345/ppo_BTC_USDT_1h_20260902_091552/)（0.5x：验证 +39.17%，测试 +8.23%，测试最大回撤 -17.95%）。加载时必须同时使用模型和同目录的 VecNormalize 文件，并保持对应 `rebalance_interval=48`、`max_leverage`、`account_leverage=2.0`。

fast/slow checkpoint 默认读取 `data/model/transformer_<symbol>_<period>_{fast,slow}.pt`；也可以通过 `RLTrainingConfig.fast_model_path` 和 `slow_model_path` 指定。

## RL 交易可视化

训练完成后，可以将每笔实际成交标注到 K 线上，并生成一个无需额外前端依赖的单文件 HTML：

```bash
coinquant report-rl data/model/rl_production/ppo_BTC_USDT_1h_20260902_074408 --split test
```

页面包含 K 线、开仓/加仓/减仓/反转/平仓标记、权益与实际暴露曲线、成交明细表和 hover 详情。滚轮可缩放，拖动 K 线可平移，点击成交明细会定位到对应 K 线。默认输出到 run 目录下的 `<split>_trades.html`，也可以用 `--output` 指定路径。

报告命令也接受 ensemble 目录，例如 `coinquant report-rl data/model/rl_ensemble_x025_v1 --split test`；此时会按 manifest 重新回放集成动作并生成 [test_trades.html](data/model/rl_ensemble_x025_v1/test_trades.html)。

训练算法可通过 `RLTrainingConfig.algorithm` 选择 `ppo` 或 `sac`。SAC 适用于连续动作探索，但在当前 BTC/USDT 1h 数据上的多种子测试换手和回撤较高，默认仍建议 PPO。`use_temporal_extractor=True` 会启用 Conv1d+GRU 时序编码器；它表达能力更强但推理较慢，默认关闭以便批量训练。

跨随机种子的 PPO 可在推理时使用动作集成。已验证的四模型 manifest 位于 [ensemble.json](data/model/rl_ensemble_x025_v1/ensemble.json)，加载方式为 `load_ensemble(...)`；manifest 同时固定各模型的 VecNormalize、权重、缩放和 0.25x 动作上限。当前 1.5x 缩放的集成在独立测试段约 +4.7%、最大回撤约 -5.6%。

如果更看重仓位和收益，可以使用 [0.5x ensemble](data/model/rl_ensemble_x05_v1/ensemble.json)。它由两个 0.5x PPO 种子集成，测试段约 +10.9%、Sharpe 0.75、最大回撤 -16.8%；由于成员间差异仍较大，建议先用 0.25x ensemble 做实盘影子运行。

## 数据处理
| 数据             | 推荐处理                      |
| -------------- | ------------------------- |
| Price(Open)    | `log` → Rolling Z-score   |
| High/Low/Close | 转成相对 Open 收益率，不再做 log     |
| Volume         | `log1p` → Rolling Z-score |
| MA | MA-slope -> Rolling Z-score |

## 回测验证指标
| 指标                   | 含义          | 为什么重要         |
| -------------------- | ----------- | ------------- |
| IC                   | 模型预测能力      | 判断alpha有没有信息量 |
| Rank IC              | 排序能力        | 比IC更稳         |
| Sharpe               | 风险调整收益      | **最重要**       |
| 年化收益（PnL）            | 最终赚钱能力      | 看绝对收益         |
| MDD                  | 最大回撤        | 控制风险          |
| Calmar               | 年化收益 ÷ 最大回撤 | 比单看PnL更有意义    |
| Turnover             | 换手率         | 估算手续费和滑点      |
| Win Rate             | 胜率          | 心理体验和稳定性参考    |
| Profit Factor        | 总盈利 ÷ 总亏损   | 判断盈亏质量        |
| Average Holding Time | 平均持仓时间      | 判断策略节奏是否合理    |
