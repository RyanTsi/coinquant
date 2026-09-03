"""Standalone HTML report for RL bar ledgers.

The report deliberately has no frontend dependency.  It embeds the selected RL
ledger and matching OHLCV bars into one HTML file that can be opened locally or
served by any static web server.  Every actual fill is marked on the candlestick
chart and is linked to the detailed trade table below it.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_TRADE_EVENTS = {"FILLED", "TARGET_REDUCED", "LIQUIDATED", "REJECTED"}


class RLTradeReport:
    """Build a self-contained trade-on-candles HTML report from an RL run."""

    def __init__(self, run_dir: str | Path, split: str = "test") -> None:
        self.run_dir = Path(run_dir)
        self.split = str(split).strip().lower()
        if self.split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid or test")

    def write(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path) if output_path else self.run_dir / f"{self.split}_trades.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path

    def render(self) -> str:
        payload = self._payload()
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
        return _HTML_TEMPLATE.replace("__RL_PAYLOAD__", encoded)

    def _payload(self) -> dict[str, Any]:
        # An ensemble directory contains a manifest rather than a single RL
        # ledger.  Evaluate the ensemble once so the page represents the
        # actual deployed action stream, not one arbitrary member.
        ensemble_manifest = self.run_dir / "ensemble.json"
        source_run_dir = self.run_dir
        if ensemble_manifest.exists():
            manifest = json.loads(ensemble_manifest.read_text(encoding="utf-8"))
            member_dirs = []
            for item in manifest.get("run_dirs", []):
                member = Path(item)
                if not member.is_absolute():
                    member = self.run_dir / member
                member_dirs.append(member)
            if not member_dirs:
                raise ValueError(f"ensemble manifest has no run_dirs: {ensemble_manifest}")
            source_run_dir = member_dirs[0]
        config_path = source_run_dir / "config.json"
        ledger_path = self.run_dir / f"{self.split}_ledger.jsonl"
        metrics_path = self.run_dir / "metrics.json"
        if not config_path.exists():
            raise FileNotFoundError(f"RL run config not found: {config_path}")
        if not ledger_path.exists() and not ensemble_manifest.exists():
            raise FileNotFoundError(f"RL ledger not found: {ledger_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if ensemble_manifest.exists() and not ledger_path.exists():
            records = self._evaluate_ensemble(source_run_dir)
        else:
            records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"RL ledger is empty: {ledger_path}")
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        metrics = dict(metrics_payload.get("metrics", {}).get(self.split, {}))
        if not metrics and ensemble_manifest.exists():
            evaluation = dict(metrics_payload.get("ensemble_evaluation", {}))
            metrics = {
                "total_return": evaluation.get("valid_return" if self.split == "valid" else "test_return"),
                "sharpe": evaluation.get("valid_sharpe" if self.split == "valid" else "test_sharpe"),
                "max_drawdown": evaluation.get("valid_max_drawdown" if self.split == "valid" else "test_max_drawdown"),
                "final_equity": 1.0 + float(evaluation.get("valid_return" if self.split == "valid" else "test_return", 0.0)),
                "trade_count": len([item for item in records if _number(item.get("turnover"), 0.0) > 1e-12]),
                "total_turnover": sum(_number(item.get("turnover"), 0.0) or 0.0 for item in records),
            }

        symbol = str(config.get("symbol", "BTC/USDT"))
        period = str(config.get("period", "1h"))
        bars = self._load_bars(symbol, period)
        bar_by_time = {int(row[2]): row for row in bars}
        previous_exposure = 0.0
        rows: list[dict[str, Any]] = []
        missing = 0
        for index, record in enumerate(records):
            timestamp = _timestamp_ms(record.get("entry_time", record.get("decision_time")))
            bar = bar_by_time.get(timestamp)
            if bar is None:
                # A ledger can be generated from a caller-supplied frame that
                # is not persisted in DuckDB.  Keep the report useful by
                # falling back to the previous decision timestamp, but count
                # the mismatch for an explicit warning in the UI.
                bar = bar_by_time.get(_timestamp_ms(record.get("decision_time")))
            if bar is None:
                missing += 1
                continue
            actual = _number(record.get("actual_exposure"), 0.0)
            previous = _number(record.get("previous_exposure"), previous_exposure)
            target = _number(record.get("target_exposure"), actual)
            event = str(record.get("execution_event_type", "MARK"))
            turnover = _number(record.get("turnover"), 0.0)
            is_trade = bool(turnover > 1e-12 and (event in _TRADE_EVENTS or record.get("action_applied", False)))
            marker = _marker(previous, actual, is_trade, event)
            rows.append(
                {
                    "index": len(rows),
                    "ledger_index": index,
                    "time": timestamp // 1000,
                    "timestamp": timestamp,
                    "datetime": _datetime_text(timestamp),
                    "open": _number(bar[3]),
                    "high": _number(bar[4]),
                    "low": _number(bar[5]),
                    "close": _number(bar[6]),
                    "volume": _number(bar[7]),
                    "action": _number(record.get("action"), target),
                    "requested_exposure": _number(record.get("requested_exposure"), target),
                    "target_exposure": target,
                    "actual_exposure": actual,
                    "previous_exposure": previous,
                    "equity": _number(record.get("equity")),
                    "drawdown": _number(record.get("drawdown")),
                    "net_return": _number(record.get("net_return")),
                    "reward": _number(record.get("reward")),
                    "turnover": turnover,
                    "fee_cost": _number(record.get("fee_cost")),
                    "slippage_cost": _number(record.get("slippage_cost")),
                    "event": event,
                    "action_applied": bool(record.get("action_applied", False)),
                    "quantity_change_ratio": _number(record.get("quantity_change_ratio")),
                    "marker": marker,
                    "liquidated": bool(record.get("liquidated", False)),
                }
            )
            previous_exposure = actual
        if not rows:
            raise ValueError("no ledger rows could be matched to OHLC bars")
        trades = [row for row in rows if row["marker"] is not None]
        return {
            "symbol": symbol,
            "period": period,
            "split": self.split,
            "run_dir": str(self.run_dir),
            "config": _jsonable(config),
            "metrics": _jsonable(metrics),
            "rows": rows,
            "trades": trades,
            "missing_rows": missing,
        }

    def _evaluate_ensemble(self, source_run_dir: Path) -> list[dict[str, Any]]:
        """Replay an ensemble manifest using the same deterministic RL env."""

        from coinquant.rl.ensemble import load_ensemble
        from coinquant.rl.env import TradingEnv
        from coinquant.rl.trainer import (
            RLTrainingConfig,
            _configs,
            _prepare_frame_for_env,
            build_predicted_frames,
        )

        config_payload = json.loads((source_run_dir / "config.json").read_text(encoding="utf-8"))
        config_payload["policy_hidden_sizes"] = tuple(config_payload.get("policy_hidden_sizes", (256, 256, 128)))
        config_payload.pop("output_dir", None)
        config = RLTrainingConfig(**{key: value for key, value in config_payload.items() if key in RLTrainingConfig.__dataclass_fields__})
        frames = build_predicted_frames(
            config.symbol,
            config.period,
            config.fast_model_path,
            config.slow_model_path,
            config.dl_vector_dim,
        )
        frames = {name: _prepare_frame_for_env(frame) for name, frame in frames.items()}
        observation_config, action_config, reward_config, env_config = _configs(config)
        env = TradingEnv(frames[self.split], observation_config, action_config, reward_config, env_config)
        agent = load_ensemble(self.run_dir / "ensemble.json", device="cpu")
        observation, _ = env.reset(seed=config.seed)
        done = False
        while not done:
            action, _ = agent.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        return [dict(item) for item in env.history]

    @staticmethod
    def _load_bars(symbol: str, period: str) -> list[tuple[Any, ...]]:
        from coinquant.datasource.database import DataBase

        rows = DataBase().query(period=period, symbol=symbol)
        if not rows:
            raise ValueError(f"no OHLCV bars found in database for {symbol} {period}")
        return rows


def render_rl_report(run_dir: str | Path, split: str = "test", output_path: str | Path | None = None) -> Path:
    """Write an RL trade report and return its path."""

    return RLTradeReport(run_dir, split).write(output_path)


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _timestamp_ms(value: Any) -> int:
    number = _number(value)
    if number is None:
        raise ValueError("ledger timestamp must be numeric")
    return int(number if abs(number) > 10_000_000_000 else number * 1000)


def _datetime_text(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat(timespec="minutes")


def _marker(previous: float, actual: float, is_trade: bool, event: str) -> dict[str, Any] | None:
    if not is_trade:
        return None
    epsilon = 1e-9
    if event == "LIQUIDATED" or actual == 0.0 and abs(previous) > epsilon:
        kind = "LIQUIDATION" if event == "LIQUIDATED" else "EXIT"
        side = "LONG" if previous > epsilon else "SHORT"
    elif abs(previous) <= epsilon:
        kind = "ENTRY"
        side = "LONG" if actual > epsilon else "SHORT" if actual < -epsilon else "FLAT"
    elif previous * actual < 0:
        kind = "REVERSAL"
        side = "LONG" if actual > 0 else "SHORT"
    elif abs(actual) > abs(previous):
        kind = "ADD"
        side = "LONG" if actual > 0 else "SHORT"
    else:
        kind = "REDUCE"
        side = "LONG" if actual > 0 else "SHORT"
    return {"kind": kind, "side": side, "label": f"{kind} {side}"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinQuant RL 交易回放</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#111b2d;--panel2:#17243a;--line:#2a3b58;--muted:#9eabc0;--text:#edf3ff;--up:#31c48d;--down:#ef6262;--long:#35d39b;--short:#ff7180;--accent:#75a7ff;--warn:#f6c453}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}.page{width:min(1500px,calc(100vw - 28px));margin:auto;padding:18px 0 36px}.header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:14px}.title h1{margin:0;font-size:24px}.sub{color:var(--muted);margin-top:4px}.controls{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end}.button,.select{background:var(--panel2);border:1px solid var(--line);color:var(--text);border-radius:6px;padding:8px 11px;cursor:pointer}.button.active{background:var(--accent);border-color:var(--accent);color:#07101f}.cards{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:9px;margin-bottom:12px}.card,.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px}.card{padding:10px 12px}.card .label{color:var(--muted);font-size:12px}.card .value{font-size:18px;font-variant-numeric:tabular-nums;margin-top:3px}.panel{padding:10px;margin-top:12px}.panel-head{display:flex;justify-content:space-between;align-items:center;margin:0 2px 8px}.panel-title{font-weight:600}.legend{display:flex;gap:11px;color:var(--muted);font-size:12px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.chart{width:100%;height:480px;display:block;touch-action:none}.chart.small{height:155px}.chart-wrap{position:relative}.tip{position:fixed;z-index:5;display:none;min-width:220px;pointer-events:none;background:#07101f;border:1px solid var(--line);border-radius:6px;padding:9px 11px;box-shadow:0 8px 28px #0008;font-size:12px}.tip strong{display:block;margin-bottom:5px}.tip-row{display:flex;justify-content:space-between;gap:22px;color:var(--muted)}.tip-row b{color:var(--text);font-weight:500}.range{display:flex;align-items:center;gap:8px;color:var(--muted);margin:3px 2px 0}.range input{width:220px;accent-color:var(--accent)}.table-wrap{overflow:auto;max-height:420px}.trades{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}.trades th,.trades td{padding:8px 9px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right}.trades th:first-child,.trades td:first-child,.trades td:nth-child(2){text-align:left}.trades th{position:sticky;top:0;background:var(--panel2);color:var(--muted);font-weight:600}.trades tbody tr{cursor:pointer}.trades tbody tr:hover{background:#1b2c48}.tag{border-radius:4px;padding:2px 6px;font-size:11px;font-weight:600}.tag.long{background:#124e42;color:#6ff0c1}.tag.short{background:#5d2635;color:#ff9aaa}.tag.exit{background:#3c4658;color:#d5deed}.tag.reversal{background:#574311;color:#ffe08b}.note{color:var(--muted);font-size:12px;margin-top:7px}@media(max-width:900px){.header{display:block}.controls{justify-content:flex-start;margin-top:10px}.cards{grid-template-columns:repeat(3,minmax(110px,1fr))}.chart{height:390px}}@media(max-width:520px){.page{width:calc(100vw - 16px);padding-top:10px}.cards{grid-template-columns:repeat(2,minmax(110px,1fr))}.chart{height:330px}.range input{width:130px}}
</style></head>
<body><main class="page">
<header class="header"><div><h1 id="title">RL 交易回放</h1><div class="sub" id="subtitle"></div></div><div class="controls"><button class="button active" data-filter="all">全部成交</button><button class="button" data-filter="entry">开仓/反转</button><button class="button" data-filter="exit">平仓/减仓</button><button class="button" id="resetView">最近 500 根</button></div></header>
<section class="cards" id="cards"></section>
<section class="panel"><div class="panel-head"><div class="panel-title">K线与实际成交位置</div><div class="legend"><span><i style="background:var(--long)"></i>多仓</span><span><i style="background:var(--short)"></i>空仓</span><span><i style="background:var(--warn)"></i>反转</span></div></div><div class="chart-wrap"><canvas id="price" class="chart"></canvas><div id="tip" class="tip"></div></div><div class="range"><span>窗口</span><input id="window" type="range"><span id="windowText"></span><span>拖动图表平移，滚轮缩放</span></div></section>
<section class="panel"><div class="panel-head"><div class="panel-title">权益 / 实际暴露</div><div class="legend"><span><i style="background:var(--accent)"></i>权益</span><span><i style="background:var(--warn)"></i>实际暴露</span></div></div><canvas id="equity" class="chart small"></canvas></section>
<section class="panel"><div class="panel-head"><div class="panel-title">实际成交明细（点击行定位 K 线）</div><div class="note" id="tradeCount"></div></div><div class="table-wrap"><table class="trades" id="tradeTable"></table></div><div class="note" id="note"></div></section>
</main><script>
const payload=__RL_PAYLOAD__;const rows=payload.rows;const tip=document.getElementById('tip');
const state={start:Math.max(0,rows.length-500),size:Math.min(500,rows.length),filter:'all',hover:null,drag:false,dragX:0,dragStart:0};
const colors={grid:'#2a3b58',axis:'#9eabc0',up:'#31c48d',down:'#ef6262',long:'#35d39b',short:'#ff7180',reversal:'#f6c453',equity:'#75a7ff',exposure:'#f6c453',hover:'#dbe8ff'};
function fmt(v,d=2){return v==null||!Number.isFinite(Number(v))?'--':Number(v).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d})}function pct(v,d=2){return v==null||!Number.isFinite(Number(v))?'--':(Number(v)*100).toFixed(d)+'%'}function visible(){return rows.slice(state.start,state.start+state.size)}function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
function setup(canvas){const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(r.width*d));canvas.height=Math.max(1,Math.round(r.height*d));const c=canvas.getContext('2d');c.setTransform(d,0,0,d,0,0);return[c,r.width,r.height]}
function rect(w,h){return{l:60,r:w-16,t:15,b:h-29}}
function yscale(min,max,a,b){const span=max-min||1;return v=>b-(v-min)/span*(b-a)}
function grid(c,r,w,h,min,max,format){c.clearRect(0,0,w,h);c.strokeStyle=colors.grid;c.fillStyle=colors.axis;c.font='11px system-ui';c.textAlign='right';c.textBaseline='middle';for(let i=0;i<=4;i++){const y=r.t+(r.b-r.t)*i/4;c.beginPath();c.moveTo(r.l,y);c.lineTo(r.r,y);c.stroke();c.fillText(format(max-(max-min)*i/4),r.l-7,y)}const vs=visible();c.textAlign='center';c.textBaseline='top';for(let i=0;i<Math.min(6,vs.length);i++){const j=Math.round((vs.length-1)*i/Math.max(1,Math.min(6,vs.length)-1));const x=r.l+(r.r-r.l)*j/Math.max(1,vs.length-1);c.fillText(vs[j].datetime.slice(5,16).replace('T',' '),x,r.b+8)}}
function markerColor(m){return m.kind==='REVERSAL'?colors.reversal:m.side==='LONG'?colors.long:colors.short}
function drawPrice(){const[c,w,h]=setup(document.getElementById('price')),vs=visible();if(!vs.length)return;const r=rect(w,h),lo=Math.min(...vs.map(x=>x.low)),hi=Math.max(...vs.map(x=>x.high)),pad=(hi-lo||hi*.01)*.08,y=yscale(lo-pad,hi+pad,r.t,r.b);grid(c,r,w,h,lo-pad,hi+pad,v=>fmt(v,0));const step=(r.r-r.l)/Math.max(1,vs.length-1),cw=Math.max(2,Math.min(11,step*.65));vs.forEach((x,i)=>{const px=r.l+step*i,up=x.close>=x.open;c.strokeStyle=up?colors.up:colors.down;c.fillStyle=c.strokeStyle;c.beginPath();c.moveTo(px,y(x.high));c.lineTo(px,y(x.low));c.stroke();c.fillRect(px-cw/2,y(Math.max(x.open,x.close)),cw,Math.max(1,y(Math.min(x.open,x.close))-y(Math.max(x.open,x.close))));const kind=x.marker&&x.marker.kind;const show=x.marker&&(state.filter==='all'||state.filter==='entry'&&['ENTRY','ADD','REVERSAL'].includes(kind)||state.filter==='exit'&&['EXIT','REDUCE','LIQUIDATION'].includes(kind));if(show){const mc=markerColor(x.marker),isUp=x.marker.side==='LONG';c.fillStyle=mc;c.strokeStyle=mc;c.lineWidth=1.5;c.beginPath();if(kind==='EXIT'||kind==='REDUCE'){c.moveTo(px,y(x.low)+10);c.lineTo(px-5,y(x.low)+2);c.lineTo(px+5,y(x.low)+2)}else{c.moveTo(px,y(x.high)-10);c.lineTo(px-5,y(x.high)-2);c.lineTo(px+5,y(x.high)-2)}c.closePath();c.fill();c.font='10px system-ui';c.textAlign='center';c.fillText(kind,px,isUp?y(x.high)-25:y(x.low)+13)}});if(state.hover!==null&&state.hover>=state.start&&state.hover<state.start+state.size){const i=state.hover-state.start,px=r.l+step*i;c.strokeStyle=colors.hover;c.beginPath();c.moveTo(px,r.t);c.lineTo(px,r.b);c.stroke()}}
function drawEquity(){const[c,w,h]=setup(document.getElementById('equity')),vs=visible();if(!vs.length)return;const r=rect(w,h),eq=vs.map(x=>x.equity),ex=vs.map(x=>x.actual_exposure),emin=Math.min(...eq),emax=Math.max(...eq),bound=Math.max(Math.abs(Math.min(...ex)),Math.abs(Math.max(...ex)),.001),y1=yscale(emin-(emax-emin||.01)*.1,emax+(emax-emin||.01)*.1,r.t,r.b),y2=yscale(-bound,bound,r.t,r.b);grid(c,r,w,h,emin-(emax-emin||.01)*.1,emax+(emax-emin||.01)*.1,v=>fmt(v,3));const step=(r.r-r.l)/Math.max(1,vs.length-1);[['equity',eq,y1],['exposure',ex,y2]].forEach(([name,vals,y])=>{c.strokeStyle=colors[name];c.lineWidth=2;c.beginPath();vals.forEach((v,i)=>{const px=r.l+step*i;if(i)c.lineTo(px,y(v));else c.moveTo(px,y(v))});c.stroke()});c.strokeStyle=colors.grid;c.setLineDash([4,4]);c.beginPath();c.moveTo(r.l,y2(0));c.lineTo(r.r,y2(0));c.stroke();c.setLineDash([])}
function redraw(){drawPrice();drawEquity();sync()}
function sync(){document.getElementById('window').value=state.size;document.getElementById('windowText').textContent=`${state.start+1}-${Math.min(rows.length,state.start+state.size)} / ${rows.length} 根`}
function setWindow(start,size){state.size=clamp(Math.round(size),Math.min(30,rows.length),rows.length);state.start=clamp(Math.round(start),0,Math.max(0,rows.length-state.size));state.hover=null;redraw()}
function renderCards(){const m=payload.metrics||{},defs=[['收益',pct(m.total_return)],['Sharpe',fmt(m.sharpe,2)],['最大回撤',pct(m.max_drawdown)],['最终权益',fmt(m.final_equity,3)],['交易次数',fmt(m.trade_count,0)],['总换手',fmt(m.total_turnover,2)]];document.getElementById('cards').innerHTML=defs.map(x=>`<div class="card"><div class="label">${x[0]}</div><div class="value">${x[1]}</div></div>`).join('')}
function renderTable(){const trades=payload.trades||[];document.getElementById('tradeCount').textContent=`共 ${trades.length} 笔实际成交` ;const head='<thead><tr><th>#</th><th>时间</th><th>类型</th><th>方向</th><th>成交后暴露</th><th>权益</th><th>换手</th><th>手续费</th><th>事件</th></tr></thead>';const body=trades.map((x,i)=>{const tag=x.marker.kind==='REVERSAL'?'reversal':x.marker.kind==='EXIT'||x.marker.kind==='REDUCE'?'exit':x.marker.side==='LONG'?'long':'short';return `<tr data-row="${x.index}"><td>${i+1}</td><td>${x.datetime}</td><td><span class="tag ${tag}">${x.marker.kind}</span></td><td>${x.marker.side}</td><td>${fmt(x.actual_exposure,4)}</td><td>${fmt(x.equity,4)}</td><td>${fmt(x.turnover,4)}</td><td>${fmt(x.fee_cost,6)}</td><td>${x.event}</td></tr>`}).join('');document.getElementById('tradeTable').innerHTML=head+'<tbody>'+body+'</tbody>';document.querySelectorAll('#tradeTable tbody tr').forEach(tr=>tr.addEventListener('click',()=>{const idx=Number(tr.dataset.row);setWindow(idx-Math.floor(state.size/2),state.size)}));document.getElementById('note').textContent=(payload.missing_rows?`有 ${payload.missing_rows} 行 ledger 无法匹配数据库 K 线。`:'')+' 点击 K 线可查看该 bar；点击交易表行可定位成交。'}
function showTip(ev){const box=ev.currentTarget.getBoundingClientRect(),r=rect(box.width,box.height),x=clamp(ev.clientX-box.left,r.l,r.r),vs=visible(),i=Math.round((x-r.l)/(r.r-r.l)*Math.max(1,vs.length-1)),idx=state.start+i;state.hover=idx;const row=rows[idx];tip.style.display='block';tip.style.left=(ev.clientX+13)+'px';tip.style.top=(ev.clientY+13)+'px';tip.innerHTML=`<strong>${row.datetime} UTC</strong><div class="tip-row"><span>OHLC</span><b>${fmt(row.open,2)} / ${fmt(row.high,2)} / ${fmt(row.low,2)} / ${fmt(row.close,2)}</b></div><div class="tip-row"><span>目标/实际暴露</span><b>${fmt(row.target_exposure,4)} / ${fmt(row.actual_exposure,4)}</b></div><div class="tip-row"><span>权益 / 回撤</span><b>${fmt(row.equity,4)} / ${pct(row.drawdown)}</b></div><div class="tip-row"><span>换手 / 事件</span><b>${fmt(row.turnover,4)} / ${row.event}</b></div>${row.marker?`<div class="tip-row"><span>成交</span><b>${row.marker.label}</b></div>`:''}`;redraw()}
function hideTip(){state.hover=null;tip.style.display='none';redraw()}
function wheel(ev){ev.preventDefault();const f=ev.deltaY>0?1.25:.8;const box=ev.currentTarget.getBoundingClientRect(),r=rect(box.width,box.height),ratio=clamp((ev.clientX-box.left-r.l)/(r.r-r.l),0,1),anchor=state.start+ratio*(state.size-1),size=clamp(Math.round(state.size*f),Math.min(30,rows.length),rows.length);setWindow(anchor-ratio*(size-1),size)}
function down(ev){state.drag=true;state.dragX=ev.clientX;state.dragStart=state.start;ev.currentTarget.setPointerCapture(ev.pointerId)}function move(ev){if(!state.drag)return;const box=ev.currentTarget.getBoundingClientRect(),r=rect(box.width,box.height),delta=(ev.clientX-state.dragX)*state.size/Math.max(1,r.r-r.l);setWindow(state.dragStart-delta,state.size)}function up(ev){state.drag=false;if(ev.currentTarget.hasPointerCapture(ev.pointerId))ev.currentTarget.releasePointerCapture(ev.pointerId)}
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.filter=b.dataset.filter;redraw()}));document.getElementById('resetView').addEventListener('click',()=>setWindow(Math.max(0,rows.length-500),Math.min(500,rows.length)));document.getElementById('window').min=Math.min(30,rows.length);document.getElementById('window').max=rows.length;document.getElementById('window').addEventListener('input',e=>setWindow(state.start,Number(e.target.value)));['price','equity'].forEach(id=>{const c=document.getElementById(id);c.addEventListener('pointermove',move);c.addEventListener('mousemove',showTip);c.addEventListener('mouseleave',hideTip);c.addEventListener('wheel',wheel,{passive:false});c.addEventListener('pointerdown',down);c.addEventListener('pointerup',up);c.addEventListener('pointercancel',up)});window.addEventListener('resize',redraw);document.getElementById('title').textContent=`${payload.symbol} ${payload.period} · RL 交易回放`;document.getElementById('subtitle').textContent=`${payload.split.toUpperCase()} · ${payload.run_dir}`;renderCards();renderTable();sync();redraw();
</script></body></html>'''
