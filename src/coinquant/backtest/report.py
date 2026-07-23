from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from coinquant.config import settings

if TYPE_CHECKING:
    from coinquant.backtest.engine import BacktestResult


class BacktestReport:
    def __init__(self, result: BacktestResult):
        self.result = result

    def write(self, output_path: str | Path | None = None) -> Path:
        path = Path(output_path) if output_path else self._default_output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path

    def render(self) -> str:
        payload = {
            "symbol": self.result.symbol,
            "period": self.result.period,
            "threshold": self.result.threshold,
            "fee_rate": self.result.fee_rate,
            "metrics": _jsonable(self.result.metrics),
            "rows": _frame_records(self.result.rows),
        }
        payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
        return _HTML_TEMPLATE.replace("__BACKTEST_PAYLOAD__", payload_json)

    def _default_output_path(self) -> Path:
        rows = self.result.rows
        start = _date_part(rows["open_datetime"].iloc[0])
        end = _date_part(rows["open_datetime"].iloc[-1])
        symbol = self.result.symbol.replace("/", "_").replace(":", "_")
        filename = f"backtest_{symbol}_{self.result.period}_{start}_{end}.html"
        return Path(settings.path.data_path) / "backtest" / filename


def _date_part(value: str) -> str:
    return value[:10].replace("-", "")


def _frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "open_time",
        "open_datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "label_close_fast",
        "label_close_slow",
        "pred_fast",
        "pred_slow",
        "next_return",
        "position_fast",
        "position_slow",
        "net_return_fast",
        "net_return_slow",
        "equity_fast",
        "equity_slow",
        "drawdown_fast",
        "drawdown_slow",
    ]
    available_columns = [column for column in columns if column in df.columns]
    clean = df.loc[:, available_columns].replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return None
    return value


_HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoinQuant Backtest</title>
<style>
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-2: #eef2f5;
  --ink: #17202a;
  --muted: #687382;
  --line: #d7dee7;
  --grid: #e7ecf2;
  --up: #118a67;
  --down: #c43d4b;
  --fast: #2563eb;
  --slow: #d97706;
  --equity: #111827;
  --shadow: 0 10px 30px rgba(25, 35, 52, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button {
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink);
  min-height: 34px;
  padding: 0 12px;
  border-radius: 6px;
  cursor: pointer;
  font: inherit;
}
button.active {
  background: #17202a;
  border-color: #17202a;
  color: white;
}
input[type="range"] {
  width: 100%;
  accent-color: #17202a;
}
.page {
  width: min(1440px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 18px 0 28px;
}
.topbar {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}
.title h1 {
  margin: 0;
  font-size: 22px;
  line-height: 1.2;
  letter-spacing: 0;
}
.title .meta {
  color: var(--muted);
  margin-top: 5px;
}
.toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  max-width: 760px;
}
.button-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  width: 100%;
}
.zoom-tools {
  display: grid;
  grid-template-columns: repeat(2, minmax(230px, 1fr));
  gap: 8px 14px;
  width: 100%;
  padding: 9px 10px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.range-field {
  display: grid;
  grid-template-columns: 42px minmax(120px, 1fr) 86px;
  gap: 8px;
  align-items: center;
  color: var(--muted);
  font-size: 12px;
}
.range-field output {
  color: var(--ink);
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(130px, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.metric {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
  box-shadow: var(--shadow);
  min-width: 0;
}
.metric-label {
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.metric-value {
  margin-top: 4px;
  font-size: 19px;
  font-weight: 700;
  line-height: 1.2;
}
.metric.fast { border-top: 3px solid var(--fast); }
.metric.slow { border-top: 3px solid var(--slow); }
.chart-stack {
  display: grid;
  gap: 10px;
}
.panel {
  position: relative;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.panel-head {
  min-height: 38px;
  padding: 9px 12px 7px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--line);
}
.panel-title {
  font-weight: 700;
}
.legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  color: var(--muted);
  font-size: 12px;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.swatch {
  width: 18px;
  height: 3px;
  border-radius: 3px;
  background: var(--ink);
}
.swatch.fast { background: var(--fast); }
.swatch.slow { background: var(--slow); }
.swatch.up { background: var(--up); }
.swatch.down { background: var(--down); }
canvas {
  display: block;
  width: 100%;
}
#priceCanvas { height: 460px; }
#predictionCanvas { height: 190px; }
#equityCanvas { height: 210px; }
.tooltip {
  position: fixed;
  z-index: 10;
  min-width: 235px;
  pointer-events: none;
  background: rgba(23, 32, 42, 0.95);
  color: #fff;
  border-radius: 8px;
  padding: 10px 12px;
  box-shadow: 0 16px 38px rgba(0,0,0,0.22);
  transform: translate(14px, 14px);
  display: none;
  font-size: 12px;
}
.tooltip-row {
  display: flex;
  justify-content: space-between;
  gap: 18px;
}
.tooltip strong {
  display: block;
  font-size: 13px;
  margin-bottom: 5px;
}
.metrics-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
  overflow: hidden;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}
.metrics-table th,
.metrics-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: right;
  white-space: nowrap;
}
.metrics-table th:first-child,
.metrics-table td:first-child {
  text-align: left;
}
.metrics-table th {
  color: var(--muted);
  font-weight: 600;
  background: var(--surface-2);
}
.metrics-table tr:last-child td {
  border-bottom: 0;
}
@media (max-width: 920px) {
  .page { width: min(100vw - 20px, 1440px); padding-top: 12px; }
  .topbar { display: block; }
  .toolbar { justify-content: flex-start; margin-top: 12px; }
  .button-row { justify-content: flex-start; }
  .zoom-tools { grid-template-columns: 1fr; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  #priceCanvas { height: 380px; }
  .metrics-table { display: block; overflow-x: auto; }
}
</style>
</head>
<body>
<main class="page">
  <section class="topbar">
    <div class="title">
      <h1 id="pageTitle">CoinQuant Backtest</h1>
      <div class="meta" id="pageMeta"></div>
    </div>
    <div class="toolbar" aria-label="chart controls">
      <div class="button-row">
        <button type="button" data-model="both" class="active">Fast + Slow</button>
        <button type="button" data-model="fast">Fast</button>
        <button type="button" data-model="slow">Slow</button>
        <button type="button" data-range="all">全部</button>
        <button type="button" data-range="500" class="active">最近 500</button>
        <button type="button" data-range="200">最近 200</button>
      </div>
      <div class="zoom-tools">
        <label class="range-field" for="windowSlider">
          <span>窗口</span>
          <input id="windowSlider" type="range" min="50" max="500" step="10">
          <output id="windowLabel"></output>
        </label>
        <label class="range-field" for="panSlider">
          <span>位置</span>
          <input id="panSlider" type="range" min="0" max="0" step="1">
          <output id="panLabel"></output>
        </label>
      </div>
    </div>
  </section>

  <section class="metric-grid" id="metricGrid"></section>

  <section class="chart-stack">
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">K线</div>
        <div class="legend">
          <span class="legend-item"><span class="swatch up"></span>上涨</span>
          <span class="legend-item"><span class="swatch down"></span>下跌</span>
        </div>
      </div>
      <canvas id="priceCanvas"></canvas>
    </div>
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">模型预测值</div>
        <div class="legend">
          <span class="legend-item"><span class="swatch fast"></span>Fast</span>
          <span class="legend-item"><span class="swatch slow"></span>Slow</span>
        </div>
      </div>
      <canvas id="predictionCanvas"></canvas>
    </div>
    <div class="panel">
      <div class="panel-head">
        <div class="panel-title">权益曲线</div>
        <div class="legend">
          <span class="legend-item"><span class="swatch fast"></span>Fast</span>
          <span class="legend-item"><span class="swatch slow"></span>Slow</span>
        </div>
      </div>
      <canvas id="equityCanvas"></canvas>
    </div>
  </section>

  <table class="metrics-table" id="metricsTable"></table>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const payload = __BACKTEST_PAYLOAD__;
const rows = payload.rows;
const minWindowSize = Math.min(50, Math.max(1, rows.length));
const defaultWindowSize = Math.min(500, Math.max(1, rows.length));
const colors = {
  grid: "#e7ecf2",
  axis: "#687382",
  ink: "#17202a",
  up: "#118a67",
  down: "#c43d4b",
  fast: "#2563eb",
  slow: "#d97706",
  equity: "#111827",
  hover: "rgba(23, 32, 42, 0.30)"
};
const state = {
  model: "both",
  range: "500",
  windowSize: defaultWindowSize,
  start: Math.max(0, rows.length - defaultWindowSize),
  end: rows.length,
  hoverIndex: null,
  dragging: false,
  dragStartX: 0,
  dragStartIndex: 0
};
const canvases = {
  price: document.getElementById("priceCanvas"),
  prediction: document.getElementById("predictionCanvas"),
  equity: document.getElementById("equityCanvas")
};
const tooltip = document.getElementById("tooltip");
const windowSlider = document.getElementById("windowSlider");
const panSlider = document.getElementById("panSlider");
const windowLabel = document.getElementById("windowLabel");
const panLabel = document.getElementById("panLabel");

function fmtNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
function fmtCompact(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function fmtPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(Number(value) * 100).toFixed(digits)}%`;
}
function fmtPred(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${Number(value).toFixed(4)}%`;
}
function activeModels() {
  if (state.model === "both") return ["fast", "slow"];
  return [state.model];
}
function visibleRows() {
  return rows.slice(state.start, state.end);
}
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
function setActiveRangeButton(range) {
  document.querySelectorAll("[data-range]").forEach(item => {
    item.classList.toggle("active", item.dataset.range === range);
  });
}
function syncZoomControls() {
  const maxStart = Math.max(0, rows.length - state.windowSize);
  windowSlider.min = String(minWindowSize);
  windowSlider.max = String(Math.max(minWindowSize, rows.length));
  windowSlider.value = String(state.windowSize);
  panSlider.min = "0";
  panSlider.max = String(maxStart);
  panSlider.value = String(state.start);
  panSlider.disabled = maxStart === 0;
  windowLabel.value = `${fmtCompact(state.windowSize)} 根`;
  panLabel.value = `${fmtCompact(state.start + 1)}-${fmtCompact(state.end)}`;
}
function setWindow(start, size) {
  const windowSize = clamp(Math.round(size), minWindowSize, rows.length);
  const maxStart = Math.max(0, rows.length - windowSize);
  state.windowSize = windowSize;
  state.start = clamp(Math.round(start), 0, maxStart);
  state.end = state.start + windowSize;
  state.hoverIndex = null;
  syncZoomControls();
  drawAll();
}
function setCustomWindow(start, size) {
  state.range = "custom";
  setActiveRangeButton(null);
  setWindow(start, size);
}
function setRange(range) {
  state.range = range;
  const size = range === "all" ? rows.length : Math.min(rows.length, Number(range));
  setActiveRangeButton(range);
  setWindow(Math.max(0, rows.length - size), size);
}
function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}
function plotRect(width, height) {
  return { left: 58, right: width - 18, top: 18, bottom: height - 30 };
}
function scaleLinear(domainMin, domainMax, rangeMin, rangeMax) {
  const span = domainMax - domainMin || 1;
  return value => rangeMax - ((value - domainMin) / span) * (rangeMax - rangeMin);
}
function drawGrid(ctx, rect, width, height, min, max, formatter) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.fillStyle = colors.axis;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const y = rect.top + (rect.bottom - rect.top) * (i / 4);
    const value = max - (max - min) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(rect.left, y);
    ctx.lineTo(rect.right, y);
    ctx.stroke();
    ctx.fillText(formatter(value), rect.left - 8, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const data = visibleRows();
  const tickCount = Math.min(6, data.length);
  for (let i = 0; i < tickCount; i += 1) {
    const index = Math.round((data.length - 1) * (i / Math.max(1, tickCount - 1)));
    const x = rect.left + (rect.right - rect.left) * (index / Math.max(1, data.length - 1));
    ctx.fillText(data[index].open_datetime.slice(5, 16), x, rect.bottom + 9);
  }
}
function drawHover(ctx, rect) {
  if (state.hoverIndex === null || state.hoverIndex < state.start || state.hoverIndex >= state.end) return;
  const localIndex = state.hoverIndex - state.start;
  const count = Math.max(1, state.end - state.start - 1);
  const x = rect.left + (rect.right - rect.left) * (localIndex / count);
  ctx.strokeStyle = colors.hover;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, rect.top);
  ctx.lineTo(x, rect.bottom);
  ctx.stroke();
}
function drawPrice() {
  const { ctx, width, height } = setupCanvas(canvases.price);
  const data = visibleRows();
  if (!data.length) return;
  const rect = plotRect(width, height);
  const lows = data.map(row => row.low);
  const highs = data.map(row => row.high);
  const min = Math.min(...lows);
  const max = Math.max(...highs);
  const padding = (max - min || max * 0.01) * 0.08;
  const y = scaleLinear(min - padding, max + padding, rect.top, rect.bottom);
  drawGrid(ctx, rect, width, height, min - padding, max + padding, value => fmtNumber(value, 0));
  const step = (rect.right - rect.left) / Math.max(1, data.length - 1);
  const candleWidth = Math.max(2, Math.min(10, step * 0.62));
  data.forEach((row, index) => {
    const x = rect.left + step * index;
    const up = row.close >= row.open;
    ctx.strokeStyle = up ? colors.up : colors.down;
    ctx.fillStyle = up ? colors.up : colors.down;
    ctx.beginPath();
    ctx.moveTo(x, y(row.high));
    ctx.lineTo(x, y(row.low));
    ctx.stroke();
    const bodyTop = y(Math.max(row.open, row.close));
    const bodyBottom = y(Math.min(row.open, row.close));
    ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, Math.max(1, bodyBottom - bodyTop));
  });
  drawHover(ctx, rect);
}
function drawLineChart(canvas, series, formatter, symmetric = false) {
  const { ctx, width, height } = setupCanvas(canvas);
  const data = visibleRows();
  if (!data.length) return;
  const rect = plotRect(width, height);
  const values = [];
  activeModels().forEach(model => {
    series(model).forEach(value => {
      if (value !== null && value !== undefined && Number.isFinite(value)) values.push(value);
    });
  });
  if (!values.length) return;
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (symmetric) {
    const bound = Math.max(Math.abs(min), Math.abs(max), 0.0001);
    min = -bound;
    max = bound;
  }
  const padding = (max - min || 1) * 0.12;
  min -= padding;
  max += padding;
  const y = scaleLinear(min, max, rect.top, rect.bottom);
  drawGrid(ctx, rect, width, height, min, max, formatter);
  if (min < 0 && max > 0) {
    ctx.strokeStyle = "#9aa5b1";
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(rect.left, y(0));
    ctx.lineTo(rect.right, y(0));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  const step = (rect.right - rect.left) / Math.max(1, data.length - 1);
  activeModels().forEach(model => {
    const valuesForModel = series(model);
    ctx.strokeStyle = colors[model];
    ctx.lineWidth = 2;
    ctx.beginPath();
    valuesForModel.forEach((value, index) => {
      const x = rect.left + step * index;
      const yy = y(value);
      if (index === 0) ctx.moveTo(x, yy);
      else ctx.lineTo(x, yy);
    });
    ctx.stroke();
  });
  drawHover(ctx, rect);
}
function drawPredictions() {
  drawLineChart(
    canvases.prediction,
    model => visibleRows().map(row => row[`pred_${model}`]),
    value => `${value.toFixed(3)}%`,
    true
  );
}
function drawEquity() {
  drawLineChart(
    canvases.equity,
    model => visibleRows().map(row => row[`equity_${model}`]),
    value => fmtNumber(value, 3),
    false
  );
}
function drawAll() {
  renderHeader();
  renderMetrics();
  renderTable();
  drawPrice();
  drawPredictions();
  drawEquity();
}
function renderHeader() {
  const first = rows[state.start];
  const last = rows[state.end - 1];
  document.getElementById("pageTitle").textContent = `${payload.symbol} ${payload.period} 回测`;
  document.getElementById("pageMeta").textContent =
    `${first.open_datetime} UTC 至 ${last.open_datetime} UTC · ${fmtCompact(state.windowSize)} 根K线 · 阈值 ${fmtPred(payload.threshold)} · 单边费率 ${fmtPct(payload.fee_rate, 3)}`;
}
function metricValue(model, key, format) {
  return format(payload.metrics[model][key]);
}
function renderMetrics() {
  const defs = [
    ["total_return", "总收益", fmtPct],
    ["sharpe", "Sharpe", value => fmtNumber(value, 2)],
    ["max_drawdown", "最大回撤", fmtPct],
    ["label_ic", "Label IC", value => fmtNumber(value, 3)],
    ["direction_accuracy", "方向准确率", fmtPct],
    ["trade_count", "交易次数", fmtCompact]
  ];
  const grid = document.getElementById("metricGrid");
  grid.innerHTML = "";
  activeModels().forEach(model => {
    defs.forEach(([key, label, format]) => {
      const item = document.createElement("div");
      item.className = `metric ${model}`;
      item.innerHTML = `<div class="metric-label">${model.toUpperCase()} · ${label}</div><div class="metric-value">${metricValue(model, key, format)}</div>`;
      grid.appendChild(item);
    });
  });
}
function renderTable() {
  const defs = [
    ["model", "模型", value => String(value).toUpperCase()],
    ["rows", "样本", fmtCompact],
    ["total_return", "总收益", fmtPct],
    ["annual_return", "年化", fmtPct],
    ["annual_volatility", "年化波动", fmtPct],
    ["sharpe", "Sharpe", value => fmtNumber(value, 2)],
    ["max_drawdown", "MDD", fmtPct],
    ["calmar", "Calmar", value => fmtNumber(value, 2)],
    ["win_rate", "胜率", fmtPct],
    ["profit_factor", "盈亏比", value => fmtNumber(value, 2)],
    ["exposure", "暴露", fmtPct],
    ["avg_holding_bars", "平均持仓", value => fmtNumber(value, 1)],
    ["label_ic", "Label IC", value => fmtNumber(value, 3)],
    ["label_rank_ic", "Rank IC", value => fmtNumber(value, 3)]
  ];
  const table = document.getElementById("metricsTable");
  const header = `<thead><tr>${defs.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead>`;
  const body = activeModels().map(model => {
    const metrics = payload.metrics[model];
    const cells = defs.map(([key, , format]) => `<td>${format(metrics[key])}</td>`).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  table.innerHTML = `${header}<tbody>${body}</tbody>`;
}
function updateTooltip(event) {
  const canvas = event.currentTarget;
  const rectBox = canvas.getBoundingClientRect();
  const rect = plotRect(rectBox.width, rectBox.height);
  const x = event.clientX - rectBox.left;
  if (x < rect.left || x > rect.right) {
    hideTooltip();
    return;
  }
  const count = Math.max(1, state.end - state.start - 1);
  const localIndex = Math.round(((x - rect.left) / (rect.right - rect.left)) * count);
  state.hoverIndex = Math.max(state.start, Math.min(state.end - 1, state.start + localIndex));
  const row = rows[state.hoverIndex];
  tooltip.style.display = "block";
  tooltip.style.left = `${event.clientX}px`;
  tooltip.style.top = `${event.clientY}px`;
  tooltip.innerHTML = `
    <strong>${row.open_datetime} UTC</strong>
    <div class="tooltip-row"><span>Open</span><span>${fmtNumber(row.open, 2)}</span></div>
    <div class="tooltip-row"><span>High</span><span>${fmtNumber(row.high, 2)}</span></div>
    <div class="tooltip-row"><span>Low</span><span>${fmtNumber(row.low, 2)}</span></div>
    <div class="tooltip-row"><span>Close</span><span>${fmtNumber(row.close, 2)}</span></div>
    <div class="tooltip-row"><span>Fast 预测</span><span>${fmtPred(row.pred_fast)}</span></div>
    <div class="tooltip-row"><span>Slow 预测</span><span>${fmtPred(row.pred_slow)}</span></div>
    <div class="tooltip-row"><span>Fast 持仓/权益</span><span>${row.position_fast} / ${fmtNumber(row.equity_fast, 3)}</span></div>
    <div class="tooltip-row"><span>Slow 持仓/权益</span><span>${row.position_slow} / ${fmtNumber(row.equity_slow, 3)}</span></div>
  `;
  drawPrice();
  drawPredictions();
  drawEquity();
}
function hideTooltip() {
  state.hoverIndex = null;
  tooltip.style.display = "none";
  drawPrice();
  drawPredictions();
  drawEquity();
}
function zoomAtPointer(event) {
  event.preventDefault();
  const rectBox = event.currentTarget.getBoundingClientRect();
  const rect = plotRect(rectBox.width, rectBox.height);
  const x = clamp(event.clientX - rectBox.left, rect.left, rect.right);
  const anchorRatio = (x - rect.left) / Math.max(1, rect.right - rect.left);
  const factor = event.deltaY > 0 ? 1.25 : 0.8;
  const nextSize = clamp(Math.round(state.windowSize * factor), minWindowSize, rows.length);
  const anchorIndex = state.start + anchorRatio * Math.max(1, state.windowSize - 1);
  const nextStart = Math.round(anchorIndex - anchorRatio * Math.max(1, nextSize - 1));
  setCustomWindow(nextStart, nextSize);
}
function beginPan(event) {
  if (event.button !== 0) return;
  state.dragging = true;
  state.dragStartX = event.clientX;
  state.dragStartIndex = state.start;
  event.currentTarget.setPointerCapture(event.pointerId);
}
function updatePan(event) {
  if (!state.dragging) return;
  const rectBox = event.currentTarget.getBoundingClientRect();
  const rect = plotRect(rectBox.width, rectBox.height);
  const barsPerPixel = state.windowSize / Math.max(1, rect.right - rect.left);
  const deltaBars = Math.round((event.clientX - state.dragStartX) * barsPerPixel);
  setCustomWindow(state.dragStartIndex - deltaBars, state.windowSize);
}
function endPan(event) {
  if (!state.dragging) return;
  state.dragging = false;
  if (event.currentTarget.hasPointerCapture(event.pointerId)) {
    event.currentTarget.releasePointerCapture(event.pointerId);
  }
}
document.querySelectorAll("[data-model]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-model]").forEach(item => item.classList.remove("active"));
    button.classList.add("active");
    state.model = button.dataset.model;
    drawAll();
  });
});
document.querySelectorAll("[data-range]").forEach(button => {
  button.addEventListener("click", () => {
    setRange(button.dataset.range);
  });
});
windowSlider.addEventListener("input", () => {
  const center = state.start + state.windowSize / 2;
  const nextSize = Number(windowSlider.value);
  setCustomWindow(Math.round(center - nextSize / 2), nextSize);
});
panSlider.addEventListener("input", () => {
  setCustomWindow(Number(panSlider.value), state.windowSize);
});
Object.values(canvases).forEach(canvas => {
  canvas.addEventListener("mousemove", updateTooltip);
  canvas.addEventListener("mouseleave", hideTooltip);
  canvas.addEventListener("wheel", zoomAtPointer, { passive: false });
  canvas.addEventListener("pointerdown", beginPan);
  canvas.addEventListener("pointermove", updatePan);
  canvas.addEventListener("pointerup", endPan);
  canvas.addEventListener("pointercancel", endPan);
});
window.addEventListener("resize", drawAll);
syncZoomControls();
drawAll();
</script>
</body>
</html>
"""
