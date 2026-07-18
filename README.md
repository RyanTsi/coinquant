# BTC 量化交易平台

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
