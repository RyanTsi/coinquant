| 数据             | 推荐处理                      |
| -------------- | ------------------------- |
| Price(Open)    | `log` → Rolling Z-score   |
| High/Low/Close | 转成相对 Open 收益率，不再做 log     |
| Volume         | `log1p` → Rolling Z-score |

