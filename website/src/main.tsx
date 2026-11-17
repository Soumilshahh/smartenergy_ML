import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
} from "recharts";
import {
  Database,
  ChartNoAxesCombined,
  Activity,
  FlaskConical,
  BookOpen,
  Code2,
  ChevronLeft,
  ChevronRight,
  Download,
  ExternalLink,
  Zap,
} from "lucide-react";
import "@fontsource/manrope/latin-400.css";
import "@fontsource/manrope/latin-500.css";
import "@fontsource/manrope/latin-600.css";
import "@fontsource/manrope/latin-700.css";
import "@fontsource/manrope/latin-800.css";
import "./styles.css";
import type { Experiment, Prediction, Metric } from "./types";

const BASE = import.meta.env.BASE_URL;
const REPO = "https://github.com/Soumilshahh/smartenergy-ml";
const initialParams = new URLSearchParams(location.search);
const n = (v: number, d = 1) =>
  v.toLocaleString("en-US", {
    maximumFractionDigits: d,
    minimumFractionDigits: d,
  });
const date = (s: string) => s.slice(0, 16).replace(" ", " · ");
const tabs = [
  { id: "forecast", name: "Forecast explorer", icon: Activity },
  { id: "data", name: "Data", icon: Database },
  { id: "models", name: "Models & insights", icon: ChartNoAxesCombined },
  { id: "errors", name: "Error laboratory", icon: FlaskConical },
  { id: "method", name: "Methodology", icon: BookOpen },
  { id: "reproduce", name: "Reproduce", icon: Code2 },
];
const axis = {
  stroke: "#617287",
  fontSize: 11,
  tickLine: false,
  axisLine: false,
};
const modelDescriptions: Record<string, string> = {
  "Training mean":
    "A constant future-hour target mean learned from the fitting period.",
  Persistence:
    "Repeat the previous six observed ten-minute intervals: sum A[t−5] through A[t].",
  "Seasonal naive":
    "Repeat the same future-hour interval from yesterday: sum A[t−143] through A[t−138].",
  Ridge:
    "A linear combination of scaled features, minimizing squared error plus an L2 coefficient penalty. Scaling makes the penalty comparable across features.",
  "Random forest":
    "Average many trees fitted to bootstrap samples and random feature subsets. Bagging reduces variance; depth and leaf size control overfitting.",
  "Gradient boosting":
    "Add shallow trees sequentially to correct residual errors. Learning rate, leaf count and L2 regularization control complexity. Early stopping is disabled to avoid a random internal holdout.",
};

function ChartFrame({
  label,
  children,
  height = 290,
  unit,
}: {
  label: string;
  children: React.ReactNode;
  height?: number;
  unit?: string;
}) {
  return (
    <>
      {unit ? <p className="axis-note">{unit}</p> : null}
      <div className="chart" role="img" aria-label={label} style={{ height }}>
        {children}
      </div>
    </>
  );
}
function EnergyTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: {
    name?: string;
    value?: number;
    color?: string;
    payload?: Prediction;
  }[];
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip">
      <strong>{String(label)}</strong>
      {payload.map((p, i) => (
        <span
          key={i}
          style={{ color: p.color === "#bd780c" ? "#8d5d13" : "#066c78" }}
        >
          {p.name}: {n(Number(p.value))} Wh
        </span>
      ))}
      <small>Each point totals its following 60 minutes.</small>
    </div>
  );
}
function ComparisonTable({
  rows,
  selected,
}: {
  rows: Metric[];
  selected: string;
}) {
  return (
    <div
      className="table-scroll"
      tabIndex={0}
      role="region"
      aria-label="Scrollable research table"
    >
      <table>
        <caption className="sr-only">
          Model comparison on aligned forecast origins
        </caption>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">MAE (Wh)</th>
            <th scope="col">RMSE (Wh)</th>
            <th scope="col">R²</th>
            <th scope="col">vs. persistence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.model}
              className={r.model === selected ? "selected-row" : ""}
            >
              <th scope="row">
                {r.model}
                {r.model === selected ? (
                  <span className="selection">Validation choice</span>
                ) : null}
              </th>
              <td>{n(r.mae)}</td>
              <td>{n(r.rmse)}</td>
              <td>{n(r.r2, 3)}</td>
              <td>
                {r.improvement_pct > 0 ? "+" : ""}
                {n(r.improvement_pct)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
function SplitTimeline({ data }: { data: Experiment }) {
  return (
    <div className="split-block">
      <div className="split-bar">
        {Object.entries(data.metadata.split).map(([k, v]) => (
          <div key={k} style={{ flex: v.n }}>
            {k}
          </div>
        ))}
      </div>
      <div className="split-details">
        {Object.entries(data.metadata.split).map(([k, v]) => (
          <div key={k}>
            <strong>
              {k} · {v.n.toLocaleString()} origins
            </strong>
            <span>{date(v.origin_start)}</span>
            <span>to {date(v.origin_end)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
function App() {
  const [data, setData] = useState<Experiment | null>(null),
    [failure, setFailure] = useState("");
  const [tab, setTab] = useState(
      tabs.some((t) => t.id === initialParams.get("view"))
        ? initialParams.get("view")!
        : "forecast",
    ),
    [model, setModel] = useState(""),
    [day, setDay] = useState(""),
    [hours, setHours] = useState(initialParams.get("hours") === "48" ? 48 : 24),
    [origin, setOrigin] = useState(
      Math.max(0, Math.min(287, Number(initialParams.get("origin")) || 0)),
    );
  const [partition, setPartition] = useState<"test" | "validation">(
      initialParams.get("partition") === "validation" ? "validation" : "test",
    ),
    [metric, setMetric] = useState<"mae" | "rmse" | "r2">(
      initialParams.get("metric") === "rmse"
        ? "rmse"
        : initialParams.get("metric") === "r2"
          ? "r2"
          : "mae",
    );
  const [period, setPeriod] = useState(
      ["day", "night"].includes(initialParams.get("period") || "")
        ? initialParams.get("period")!
        : "all",
    ),
    [demand, setDemand] = useState(
      ["low", "medium", "high"].includes(initialParams.get("demand") || "")
        ? initialParams.get("demand")!
        : "all",
    ),
    [edaView, setEdaView] = useState(
      ["hourly", "distribution"].includes(initialParams.get("dataView") || "")
        ? initialParams.get("dataView")!
        : "daily",
    ),
    [dataMonth, setDataMonth] = useState(initialParams.get("month") || "all");
  useEffect(() => {
    const controller = new AbortController();
    fetch(BASE + "research/experiment.json", { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error(`Artifacts unavailable (${r.status})`);
        return r.json();
      })
      .then((d: Experiment) => {
        setData(d);
        setModel(
          d.metrics.test.some((r) => r.model === initialParams.get("model"))
            ? initialParams.get("model")!
            : d.metadata.selected_ml,
        );
        setDay(
          d.predictions.some((p) =>
            p.origin.startsWith(initialParams.get("day") || "invalid"),
          )
            ? initialParams.get("day")!
            : d.predictions[0].origin.slice(0, 10),
        );
        if (
          !d.eda.daily.some((r) =>
            r.date.startsWith(initialParams.get("month") || "invalid"),
          )
        )
          setDataMonth("all");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setFailure(e.message);
      });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!data) return;
    const query = new URLSearchParams({
      view: tab,
      model,
      day,
      hours: String(hours),
      origin: String(origin),
      partition,
      metric,
      period,
      demand,
      dataView: edaView,
      month: dataMonth,
    });
    history.replaceState(null, "", `${location.pathname}?${query}`);
  }, [
    data,
    tab,
    model,
    day,
    hours,
    origin,
    partition,
    metric,
    period,
    demand,
    edaView,
    dataMonth,
  ]);
  const days = useMemo(
    () =>
      data
        ? [...new Set(data.predictions.map((p) => p.origin.slice(0, 10)))]
        : [],
    [data],
  );
  const visible = useMemo(() => {
    if (!data) return [];
    const start = data.predictions.findIndex(
      (p) => p.origin.slice(0, 10) === day,
    );
    return data.predictions.slice(
      Math.max(0, start),
      Math.max(0, start) + hours * 6,
    );
  }, [data, day, hours]);
  const errorRows = useMemo(() => {
    if (!data) return [];
    const { low, high } = data.metadata.thresholds_train_only;
    return data.predictions
      .filter((p) => {
        const h = Number(p.origin.slice(11, 13));
        const a = p.actual;
        return (
          (period === "all" ||
            (period === "day" ? h >= 7 && h < 22 : h < 7 || h >= 22)) &&
          (demand === "all" ||
            (demand === "low"
              ? a <= low
              : demand === "high"
                ? a >= high
                : a > low && a < high))
        );
      })
      .map((p) => ({
        ...p,
        error: Number(p[model]) - p.actual,
        predicted: Number(p[model]),
      }));
  }, [data, model, period, demand]);
  if (failure)
    return (
      <main className="loading">
        <h1>Research data could not load</h1>
        <p>{failure}. Check your connection and reload the page.</p>
        <button onClick={() => location.reload()}>Reload research</button>
      </main>
    );
  if (!data)
    return (
      <main className="loading" aria-live="polite">
        <Zap size={32} />
        <h1>Loading the energy lab…</h1>
        <p>Reading verified experiment artifacts.</p>
      </main>
    );
  const p =
    visible[Math.min(origin, visible.length - 1)] || data.predictions[0];
  const models = data.metrics.test.map((r) => r.model);
  const primary = data.metrics.test.find(
    (r) => r.model === data.metadata.selected_ml,
  )!;
  const baseline = data.metrics.test.find((r) => r.model === "Persistence")!;
  const validationWinner = data.metrics.validation.find(
    (r) => r.model === data.metadata.selected_model,
  )!;
  const rows = data.metrics[partition];
  const goTab = (id: string) => {
    setTab(id);
    window.scrollTo({ top: 0, behavior: "instant" });
    document.getElementById("main")?.focus({ preventScroll: true });
  };
  const modelControl = (
    <label>
      Model
      <select
        aria-label="Model"
        value={model}
        onChange={(e) => setModel(e.target.value)}
      >
        {models.map((m) => (
          <option key={m}>{m}</option>
        ))}
      </select>
    </label>
  );
  const filteredMae = errorRows.length
    ? errorRows.reduce((s, r) => s + Math.abs(r.error), 0) / errorRows.length
    : 0;
  const residualBins = Array.from({ length: 20 }, (_, i) => {
    const min = -2000 + i * 200;
    return {
      error: min + 100,
      count: errorRows.filter(
        (r) =>
          r.error >= min && (i === 19 ? r.error <= 2000 : r.error < min + 200),
      ).length,
    };
  });
  return (
    <div className="app-shell">
      <a className="skip" href="#main">
        Skip to research
      </a>
      <aside className="sidebar">
        <a className="brand" href="#main" onClick={() => goTab("forecast")}>
          <Zap aria-hidden="true" size={23} fill="currentColor" />
          SmartEnergy
        </a>
        <div className="sidebar-intro">
          <strong>Energy Forecasting Lab</strong>
          <p>
            Classical machine learning.
            <br />
            One household. One hour ahead.
          </p>
        </div>
        <nav aria-label="Research sections">
          {tabs.map((t) => (
            <a
              key={t.id}
              href={`?${new URLSearchParams({ ...Object.fromEntries(new URLSearchParams(location.search)), view: t.id })}`}
              aria-current={tab === t.id ? "page" : undefined}
              onClick={(e) => {
                if (!e.metaKey && !e.ctrlKey && !e.shiftKey && e.button === 0) {
                  e.preventDefault();
                  goTab(t.id);
                }
              }}
            >
              <t.icon size={19} aria-hidden="true" />
              {t.name}
            </a>
          ))}
        </nav>
        <div className="sidebar-footer">
          <strong>Soumil Shah</strong>
          <span>ES 335 · Individual project</span>
          <a href={REPO} target="_blank" rel="noreferrer">
            Source repository <ExternalLink size={13} />
          </a>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div>
            <strong>Energy Forecasting Lab</strong>
            <span>SmartEnergy research workspace</span>
          </div>
          <span className="offline">
            <span aria-hidden="true" />
            Offline historical evaluation
          </span>
        </header>
        <main id="main" tabIndex={-1}>
          {tab === "forecast" ? (
            <>
              <section className="page-heading">
                <h1>Explore the next hour.</h1>
                <p>Past measurements in. Next-hour energy out.</p>
                <p className="research-question">
                  How accurately can classical ML forecast the next 60 minutes
                  of combined appliance energy from historical consumption, time
                  and household sensors?
                </p>
              </section>
              <div className="finding">
                <ChartNoAxesCombined size={21} aria-hidden="true" />
                <p>
                  <strong>{data.metadata.selected_ml}</strong> achieved{" "}
                  <strong>{n(primary.mae)} Wh MAE</strong> on the final test
                  period:{" "}
                  <strong>
                    {n(Math.abs(primary.improvement_pct))}%{" "}
                    {primary.improvement_pct >= 0 ? "lower" : "higher"}
                  </strong>{" "}
                  error than persistence
                  {primary.improvement_pct < 0
                    ? " (signed improvement shown below)"
                    : ""}
                  . The model was chosen using validation data.
                </p>
              </div>
              <div className="forecast-layout">
                <div>
                  <div className="controls">
                    {modelControl}
                    <label>
                      Date
                      <select
                        aria-label="Forecast date"
                        value={day}
                        onChange={(e) => {
                          setDay(e.target.value);
                          setOrigin(0);
                        }}
                      >
                        {days.map((d) => (
                          <option key={d}>{d}</option>
                        ))}
                      </select>
                    </label>
                    <fieldset>
                      <legend>Range</legend>
                      <div className="segmented">
                        {[24, 48].map((h) => (
                          <button
                            key={h}
                            aria-pressed={hours === h}
                            onClick={() => {
                              setHours(h);
                              setOrigin(0);
                            }}
                          >
                            {h}h
                          </button>
                        ))}
                      </div>
                    </fieldset>
                  </div>
                  <ChartFrame
                    label="Actual and predicted next-hour energy in Wh by forecast origin"
                    unit="Next-hour energy (Wh)"
                    height={310}
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={visible}
                        margin={{ top: 15, right: 12, bottom: 10, left: 0 }}
                      >
                        <CartesianGrid stroke="#e8edf1" vertical={false} />
                        <XAxis
                          dataKey="origin"
                          {...axis}
                          minTickGap={55}
                          tickFormatter={(v) => String(v).slice(11, 16)}
                        />
                        <YAxis {...axis} width={50} />
                        <Tooltip content={<EnergyTooltip />} />
                        <Line
                          type="linear"
                          dataKey="actual"
                          name="Actual"
                          stroke="#bd780c"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                        <Line
                          type="linear"
                          dataKey={model}
                          name={model}
                          stroke="#087f8c"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                        {model !== "Persistence" ? (
                          <Line
                            dataKey="Persistence"
                            name="Persistence"
                            stroke="#9ca8b7"
                            strokeDasharray="4 4"
                            dot={false}
                            isAnimationActive={false}
                          />
                        ) : null}
                        <ReferenceLine
                          x={p.origin}
                          stroke="#24354b"
                          strokeDasharray="3 3"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartFrame>
                  <div className="chart-legend">
                    <span className="actual">Actual future-hour total</span>
                    <span className="predicted">{model}</span>
                    {model !== "Persistence" ? (
                      <span className="baseline">Persistence</span>
                    ) : null}
                  </div>
                  <p className="caption">
                    Horizontal axis: forecast origin in the dataset clock. Each
                    point covers its following hour; neighboring targets
                    overlap.
                  </p>
                  <div className="scrubber">
                    <label htmlFor="origin">Forecast origin</label>
                    <input
                      id="origin"
                      type="range"
                      min={0}
                      max={visible.length - 1}
                      value={Math.min(origin, visible.length - 1)}
                      onChange={(e) => setOrigin(Number(e.target.value))}
                    />
                    <button
                      aria-label="Previous forecast"
                      disabled={origin === 0}
                      onClick={() => setOrigin((v) => Math.max(0, v - 1))}
                    >
                      <ChevronLeft size={18} />
                    </button>
                    <button
                      aria-label="Next forecast"
                      disabled={origin >= visible.length - 1}
                      onClick={() =>
                        setOrigin((v) => Math.min(visible.length - 1, v + 1))
                      }
                    >
                      <ChevronRight size={18} />
                    </button>
                  </div>
                </div>
                <aside className="explanation">
                  <h2>What does this view show?</h2>
                  <p>
                    A genuine held-out forecast at a historical origin, using
                    only information available by that time.
                  </p>
                  <h3>Forecast origin (t)</h3>
                  <p>
                    {date(p.origin)}
                    <br />
                    The current ten-minute energy reading is assumed fully
                    observed.
                  </p>
                  <h3>Historical window</h3>
                  <p>
                    Past energy up to 24 hours, calendar features and current
                    household sensor averages.
                  </p>
                  <h3>Future target (t + 10 to t + 60)</h3>
                  <p>
                    Sum the next six readings. The unit is{" "}
                    <strong>Wh of energy</strong>, not watts of power.
                  </p>
                </aside>
              </div>
              <div className="readouts" aria-live="polite">
                <div>
                  <span>Predicted next hour</span>
                  <strong data-testid="prediction">
                    {n(Number(p[model]))}
                    <small> Wh</small>
                  </strong>
                  <span>{model}</span>
                </div>
                <div>
                  <span>Observed next hour</span>
                  <strong>
                    {n(p.actual)}
                    <small> Wh</small>
                  </strong>
                  <span>
                    {date(p.target_start)}–{p.target_end.slice(11, 16)}
                  </span>
                </div>
                <div>
                  <span>Signed forecast error</span>
                  <strong className="error-number">
                    {n(Number(p[model]) - p.actual)}
                    <small> Wh</small>
                  </strong>
                  <span>
                    Prediction − actual · |error|{" "}
                    {n(Math.abs(Number(p[model]) - p.actual))} Wh
                  </span>
                </div>
              </div>
              <section className="section">
                <div className="section-title">
                  <h2>Model comparison</h2>
                  <button
                    className="text-button"
                    onClick={() => goTab("models")}
                  >
                    Inspect methods & validation <ChevronRight size={16} />
                  </button>
                </div>
                <ComparisonTable
                  rows={data.metrics.test}
                  selected={data.metadata.selected_model}
                />
                <p className="caption">
                  Final test results on {primary.n.toLocaleString()} common
                  origins. Positive percentages mean lower MAE than persistence.
                </p>
              </section>
              <section className="section">
                <h2>Chronological split</h2>
                <SplitTimeline data={data} />
              </section>
            </>
          ) : null}
          {tab === "data" ? (
            <>
              <section className="page-heading">
                <h1>A household, measured over time.</h1>
                <p>
                  One row represents one ten-minute interval in a low-energy
                  Belgian house.
                </p>
              </section>
              <div className="readouts">
                <div>
                  <span>Recorded observations</span>
                  <strong>{data.audit.rows.toLocaleString()}</strong>
                  <span>{data.audit.columns} source columns</span>
                </div>
                <div>
                  <span>Measurement cadence</span>
                  <strong>
                    10<small> minutes</small>
                  </strong>
                  <span>
                    {data.audit.start} to {data.audit.end}
                  </span>
                </div>
                <div>
                  <span>Timestamp gaps / missing cells</span>
                  <strong>
                    {data.audit.irregular_intervals} /{" "}
                    {data.audit.missing_total}
                  </strong>
                  <span>
                    {data.audit.duplicate_timestamps} duplicate timestamps
                  </span>
                </div>
              </div>
              <section className="section">
                <div className="section-title">
                  <h2>Consumption patterns</h2>
                  <label>
                    View
                    <select
                      aria-label="Data view"
                      value={edaView}
                      onChange={(e) => setEdaView(e.target.value)}
                    >
                      <option value="daily">Daily total</option>
                      <option value="hourly">Hour-of-day profile</option>
                      <option value="distribution">
                        Ten-minute distribution
                      </option>
                    </select>
                  </label>
                </div>
                {edaView === "daily" ? (
                  <>
                    <label className="month-control">
                      Period
                      <select
                        aria-label="Data period"
                        value={dataMonth}
                        onChange={(e) => setDataMonth(e.target.value)}
                      >
                        <option value="all">All recorded months</option>
                        {[
                          ...new Set(
                            data.eda.daily.map((r) => r.date.slice(0, 7)),
                          ),
                        ].map((month) => (
                          <option key={month}>{month}</option>
                        ))}
                      </select>
                    </label>
                    <ChartFrame
                      label="Daily total appliance energy in kWh"
                      unit="Daily energy (kWh)"
                    >
                      <ResponsiveContainer>
                        <AreaChart
                          data={data.eda.daily.filter(
                            (r) =>
                              dataMonth === "all" ||
                              r.date.startsWith(dataMonth),
                          )}
                        >
                          <CartesianGrid vertical={false} stroke="#e8edf1" />
                          <XAxis {...axis} dataKey="date" minTickGap={70} />
                          <YAxis {...axis} />
                          <Tooltip
                            formatter={(v) => [
                              n(Number(v), 2) + " kWh",
                              "Daily energy",
                            ]}
                          />
                          <Area
                            dataKey="kwh"
                            stroke="#087f8c"
                            fill="#e3f1f3"
                            isAnimationActive={false}
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                    <p className="caption">
                      Use the chart tooltip to inspect each date. Boundary days
                      are partial. This view is a descriptive full-dataset
                      summary, never a training input.
                    </p>
                  </>
                ) : (
                  <ChartFrame
                    label={
                      edaView === "hourly"
                        ? "Mean ten-minute Wh by hour of day"
                        : "Ten-minute Wh distribution"
                    }
                  >
                    <ResponsiveContainer>
                      <BarChart
                        data={
                          edaView === "hourly"
                            ? data.eda.hourly.map((r) => ({
                                x: r.hour,
                                value: r.wh,
                              }))
                            : data.eda.distribution.map((r) => ({
                                x: r.wh,
                                value: r.count,
                              }))
                        }
                      >
                        <CartesianGrid vertical={false} stroke="#e8edf1" />
                        <XAxis {...axis} dataKey="x" minTickGap={25} />
                        <YAxis {...axis} />
                        <Tooltip />
                        <Bar
                          dataKey="value"
                          fill="#087f8c"
                          isAnimationActive={false}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartFrame>
                )}
              </section>
              <div className="two-column">
                <section>
                  <h2>Quality & scope</h2>
                  <p>
                    The target channel measures{" "}
                    <strong>combined appliance energy</strong>. It cannot
                    identify individual devices. No high readings were removed:
                    these may be real activity.
                  </p>
                  <p>
                    There are {data.audit.zero_energy_count} zero readings. The
                    median is {n(data.audit.energy_wh["50%"], 0)} Wh per
                    ten-minute interval; the 99th percentile is{" "}
                    {n(data.audit.energy_wh["99%"], 0)} Wh. Negative-energy and
                    invalid-humidity checks found {data.audit.negative_energy}{" "}
                    and {data.audit.invalid_humidity} records respectively.
                  </p>
                  <p>
                    Weekday mean: {n(data.eda.weekday_weekend.weekday)} Wh.
                    Weekend mean: {n(data.eda.weekday_weekend.weekend)} Wh.
                    These are observational summaries over a limited period.
                  </p>
                </section>
                <section>
                  <h2>Timing deserves care</h2>
                  <p>
                    UCI describes airport weather as interpolated hourly data.
                    Interpolation may use a future endpoint, so airport fields
                    are excluded. Random controls rv1 and rv2 are also excluded.
                  </p>
                  <p>
                    Household temperature and humidity averages at t are assumed
                    available when interval t ends. Source timestamps have no
                    timezone offset; the lab displays them unchanged.
                  </p>
                  <p>
                    Adjacent future-hour targets have correlation{" "}
                    {n(data.eda.autocorrelation[0].correlation, 3)}. Their
                    shared readings make neighboring errors dependent.
                  </p>
                </section>
              </div>
              <section className="section">
                <h2>Research figures</h2>
                <div className="figure-grid">
                  {data.eda.figures.map((f) => (
                    <figure key={f.file}>
                      <a
                        href={BASE + "research/figures/" + f.file}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <img
                          loading="lazy"
                          width="800"
                          height="380"
                          src={BASE + "research/figures/" + f.file}
                          alt={f.title}
                        />
                      </a>
                      <figcaption>{f.takeaway}</figcaption>
                    </figure>
                  ))}
                </div>
              </section>
            </>
          ) : null}
          {tab === "models" ? (
            <>
              <section className="page-heading">
                <h1>Earn the extra complexity.</h1>
                <p>
                  Three classical ML candidates, three forecasting rules,
                  identical origins.
                </p>
              </section>
              <div className="controls compact">
                <label>
                  Evaluation period
                  <select
                    aria-label="Evaluation period"
                    value={partition}
                    onChange={(e) =>
                      setPartition(e.target.value as typeof partition)
                    }
                  >
                    <option value="test">Final untouched test</option>
                    <option value="validation">Validation (selection)</option>
                  </select>
                </label>
                <label>
                  Comparison metric
                  <select
                    aria-label="Comparison metric"
                    value={metric}
                    onChange={(e) => setMetric(e.target.value as typeof metric)}
                  >
                    <option value="mae">MAE · Wh · lower is better</option>
                    <option value="rmse">RMSE · Wh · lower is better</option>
                    <option value="r2">R² · higher is better</option>
                  </select>
                </label>
              </div>
              <ChartFrame
                label={`${partition} comparison by ${metric}`}
                height={270}
              >
                <ResponsiveContainer>
                  <BarChart
                    data={rows}
                    layout="vertical"
                    margin={{ left: 25, right: 30 }}
                  >
                    <CartesianGrid horizontal={false} stroke="#e8edf1" />
                    <XAxis type="number" {...axis} />
                    <YAxis
                      type="category"
                      dataKey="model"
                      {...axis}
                      width={125}
                    />
                    <Tooltip
                      formatter={(v) => n(Number(v), metric === "r2" ? 3 : 1)}
                    />
                    <Bar
                      dataKey={metric}
                      fill="#087f8c"
                      radius={[0, 3, 3, 0]}
                      isAnimationActive={false}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </ChartFrame>
              <ComparisonTable
                rows={rows}
                selected={data.metadata.selected_model}
              />
              <div className="two-column section">
                <section>
                  <h2>What selection actually used</h2>
                  <p>
                    <strong>{data.metadata.selected_model}</strong> had the
                    lowest validation MAE, {n(validationWinner.mae)} Wh.
                    Hyperparameters were chosen by three expanding training-only
                    folds with a six-origin gap. The primary feature set was
                    declared in advance.
                  </p>
                  <p>
                    All frozen candidates were refitted on pre-test data and
                    reported for transparency. Final test rankings never changed
                    the selected method.
                  </p>
                  <p>
                    Random Forest has the lowest final-test MAE in this
                    comparison. We retain the validation-selected model:
                    choosing a new winner from the final test would reuse it for
                    selection.
                  </p>
                  <p>
                    Selected ML test MAE: {n(primary.mae)} Wh; persistence:{" "}
                    {n(baseline.mae)} Wh. Relative MAE reduction:{" "}
                    {n(primary.improvement_pct)}%.
                  </p>
                </section>
                <section>
                  <h2>How strong is the evidence?</h2>
                  <p>
                    A paired calendar-day bootstrap gives a 95% interval of{" "}
                    <strong>
                      {n(data.metrics.bootstrap.ci95_pct[0])}% to{" "}
                      {n(data.metrics.bootstrap.ci95_pct[1])}%
                    </strong>{" "}
                    for the selected ML model’s MAE reduction over persistence (
                    {data.metrics.bootstrap.repetitions.toLocaleString()}{" "}
                    resamples, {data.metrics.bootstrap.days} calendar-day
                    blocks).
                  </p>
                  <p>
                    {data.metrics.bootstrap.ci95_pct[0] > 0
                      ? "The interval favors the ML model in this test record."
                      : "The interval includes no improvement; the evidence does not establish a reliable gain."}{" "}
                    Across-day dependence, boundary days and one-household
                    sampling limit this interval.
                  </p>
                  <p>
                    On origins spaced one hour apart, selected-model MAE is{" "}
                    {n(
                      data.metrics.nonoverlapping_test.find(
                        (r) => r.model === data.metadata.selected_ml,
                      )!.mae,
                    )}{" "}
                    Wh.
                  </p>
                </section>
              </div>
              <section className="section">
                <h2>Do sensors add predictive value?</h2>
                <p>
                  Validation ablation using {data.metadata.selected_ml}, the
                  same frozen hyperparameters and the same samples. This is a
                  controlled feature comparison; each group was not separately
                  retuned.
                </p>
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Scrollable research table"
                >
                  <table>
                    <thead>
                      <tr>
                        <th>Feature group</th>
                        <th>Features</th>
                        <th>Validation MAE (Wh)</th>
                        <th>RMSE (Wh)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.ablation.map((r) => (
                        <tr key={r.group}>
                          <th scope="row">{r.group}</th>
                          <td>{r.features}</td>
                          <td>{n(r.mae)}</td>
                          <td>{n(r.rmse)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>
                  Adding sensors{" "}
                  {data.ablation[2].mae < data.ablation[1].mae
                    ? "reduced"
                    : "increased"}{" "}
                  validation MAE by{" "}
                  {n(Math.abs(data.ablation[2].mae - data.ablation[1].mae))} Wh.
                  Correlated or drifting sensors can add noise; this is
                  predictive evidence, not a causal claim.
                </p>
              </section>
              <section className="section">
                <h2>What the model relies on</h2>
                <p>
                  Validation permutation importance: increase in MAE when one
                  feature is shuffled, three repeats. Correlated features can
                  share or mask importance; temporal shuffling can create
                  unrealistic combinations.
                </p>
                <ChartFrame
                  label="Top validation permutation feature importances in MAE Wh"
                  height={340}
                >
                  <ResponsiveContainer>
                    <BarChart
                      data={data.importance.slice(0, 10)}
                      layout="vertical"
                      margin={{ left: 20, right: 20 }}
                    >
                      <CartesianGrid horizontal={false} stroke="#e8edf1" />
                      <XAxis {...axis} type="number" />
                      <YAxis
                        {...axis}
                        type="category"
                        dataKey="feature"
                        width={155}
                      />
                      <Tooltip formatter={(v) => n(Number(v)) + " Wh"} />
                      <Bar
                        dataKey="mae_increase"
                        fill="#087f8c"
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartFrame>
              </section>
              <section className="section">
                <h2>The methods, explained</h2>
                <div className="method-list">
                  {models.map((m) => (
                    <details key={m}>
                      <summary>{m}</summary>
                      <p>{modelDescriptions[m]}</p>
                      {data.metadata.best_parameters[m] ? (
                        <pre tabIndex={0}>
                          {JSON.stringify(
                            data.metadata.best_parameters[m],
                            null,
                            2,
                          )}
                        </pre>
                      ) : null}
                    </details>
                  ))}
                </div>
              </section>
            </>
          ) : null}
          {tab === "errors" ? (
            <>
              <section className="page-heading">
                <h1>Where forecasts fall short.</h1>
                <p>
                  Inspect sudden activity, quiet periods and the cost of missed
                  demand.
                </p>
              </section>
              <div className="controls">
                {modelControl}
                <label>
                  Time of day
                  <select
                    aria-label="Time of day"
                    value={period}
                    onChange={(e) => setPeriod(e.target.value)}
                  >
                    <option value="all">All hours</option>
                    <option value="day">Daytime · 07–22</option>
                    <option value="night">Nighttime · 22–07</option>
                  </select>
                </label>
                <label>
                  Consumption
                  <select
                    aria-label="Consumption"
                    value={demand}
                    onChange={(e) => setDemand(e.target.value)}
                  >
                    <option value="all">All demand</option>
                    <option value="low">Low demand</option>
                    <option value="medium">Medium demand</option>
                    <option value="high">High demand</option>
                  </select>
                </label>
              </div>
              <p
                className="filter-summary"
                aria-live="polite"
                data-testid="filter-summary"
              >
                {errorRows.length.toLocaleString()} origins · MAE{" "}
                {n(filteredMae)} Wh · {model}
              </p>
              <p className="caption">
                Training-only thresholds: low ≤{" "}
                {n(data.metadata.thresholds_train_only.low, 0)} Wh; high ≥{" "}
                {n(data.metadata.thresholds_train_only.high, 0)} Wh. These
                outcome-based groups are retrospective diagnostics.
              </p>
              {errorRows.length ? (
                <div className="two-column">
                  <section>
                    <h2>Actual versus predicted</h2>
                    <ChartFrame label="Scatter of actual and predicted future-hour Wh">
                      <ResponsiveContainer>
                        <ScatterChart
                          margin={{ bottom: 20, left: 5, right: 15 }}
                        >
                          <CartesianGrid stroke="#e8edf1" />
                          <XAxis
                            {...axis}
                            type="number"
                            dataKey="actual"
                            name="Actual"
                            label={{
                              value: "Actual (Wh)",
                              position: "insideBottom",
                              offset: -12,
                            }}
                          />
                          <YAxis
                            {...axis}
                            type="number"
                            dataKey="predicted"
                            name="Predicted"
                            width={55}
                          />
                          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                          <Scatter
                            data={errorRows}
                            fill="#087f8c"
                            fillOpacity={0.35}
                            isAnimationActive={false}
                          />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                    <p className="caption">
                      All filtered origins are plotted. Vertical axis: predicted
                      Wh. Each dot is one overlapping future-hour target.
                    </p>
                  </section>
                  <section>
                    <h2>Signed error distribution</h2>
                    <ChartFrame label="Histogram of prediction minus actual energy Wh">
                      <ResponsiveContainer>
                        <BarChart data={residualBins}>
                          <CartesianGrid vertical={false} stroke="#e8edf1" />
                          <XAxis {...axis} dataKey="error" minTickGap={30} />
                          <YAxis {...axis} />
                          <Tooltip />
                          <Bar
                            dataKey="count"
                            fill="#087f8c"
                            isAnimationActive={false}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartFrame>
                    <p className="caption">
                      Prediction − actual (Wh). Negative errors miss demand;
                      positive errors overpredict. Range −2,000 to +2,000 Wh;{" "}
                      {
                        errorRows.filter(
                          (r) => r.error < -2000 || r.error > 2000,
                        ).length
                      }{" "}
                      outside-range errors are excluded from this histogram
                      only.
                    </p>
                  </section>
                </div>
              ) : (
                <p role="status">
                  No origins match these filters. Select a broader time or
                  consumption group.
                </p>
              )}
              <section className="section">
                <h2>Largest errors in this selection</h2>
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Scrollable research table"
                >
                  <table>
                    <thead>
                      <tr>
                        <th>Forecast origin</th>
                        <th>Actual (Wh)</th>
                        <th>Predicted (Wh)</th>
                        <th>Error (Wh)</th>
                        <th>Inspect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...errorRows]
                        .sort((a, b) => Math.abs(b.error) - Math.abs(a.error))
                        .slice(0, 5)
                        .map((r) => (
                          <tr key={r.origin}>
                            <th scope="row">{date(r.origin)}</th>
                            <td>{n(r.actual)}</td>
                            <td>{n(r.predicted)}</td>
                            <td>{n(r.error)}</td>
                            <td>
                              <button
                                className="text-button"
                                onClick={() => {
                                  setDay(r.origin.slice(0, 10));
                                  setHours(24);
                                  const start = data.predictions.findIndex(
                                    (p) =>
                                      p.origin.slice(0, 10) ===
                                      r.origin.slice(0, 10),
                                  );
                                  setOrigin(
                                    data.predictions.findIndex(
                                      (p) => p.origin === r.origin,
                                    ) - start,
                                  );
                                  goTab("forecast");
                                }}
                              >
                                Open forecast
                              </button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
                <p>
                  These cases were selected by absolute error for diagnosis, not
                  for model selection. Sudden appliance starts may be difficult
                  to infer from past readings; occupant intent is unobserved, so
                  that explanation remains a hypothesis.
                </p>
              </section>
              <section className="section">
                <h2>Selected-model failure audit</h2>
                <p>
                  Frozen model: {data.metadata.selected_ml}. Sudden-change
                  threshold |future total − persistence| ≥{" "}
                  {n(data.metadata.thresholds_train_only.sudden_change, 0)} Wh,
                  defined from training targets.
                </p>
                <div
                  className="table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Scrollable research table"
                >
                  <table>
                    <thead>
                      <tr>
                        <th>Condition</th>
                        <th>Origins</th>
                        <th>MAE (Wh)</th>
                        <th>RMSE (Wh)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.errors.map((r) => (
                        <tr key={r.group}>
                          <th scope="row">{r.group}</th>
                          <td>{r.n}</td>
                          <td>{n(r.mae)}</td>
                          <td>{n(r.rmse)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          ) : null}
          {tab === "method" ? (
            <>
              <section className="page-heading">
                <h1>A forecast you can audit.</h1>
                <p>
                  Short-Term Household Appliance Energy Consumption Forecasting
                  Using Machine Learning
                </p>
              </section>
              <div className="formula">
                <span>At the end of interval t</span>
                <strong>Y(t) = A(t+1) + A(t+2) + … + A(t+6)</strong>
                <p>
                  Six ten-minute readings sum to next-hour energy in Wh. Divide
                  Wh by 1,000 for kWh. W measures power; it is not the target.
                </p>
              </div>
              <section className="section">
                <h2>From observations to evidence</h2>
                <ol className="pipeline">
                  <li>
                    <strong>Verify data</strong>
                    <span>
                      Official UCI CSV; check timestamps, units, missing cells
                      and physical ranges. Fail closed on irregular intervals.
                    </span>
                  </li>
                  <li>
                    <strong>Build past-only features</strong>
                    <span>
                      Energy lags up to 24h; trailing means, standard
                      deviations, minima, maxima, totals and trends. Windows
                      include t.
                    </span>
                  </li>
                  <li>
                    <strong>Partition chronologically</strong>
                    <span>
                      Approximately 60% train, 20% validation, 20% test. Purge
                      six origins before each boundary.
                    </span>
                  </li>
                  <li>
                    <strong>Tune and validate</strong>
                    <span>
                      Three expanding training folds with gap=6 select
                      hyperparameters; validation MAE selects the final method.
                    </span>
                  </li>
                  <li>
                    <strong>Freeze, refit, test</strong>
                    <span>
                      Freeze selection before test access. Refit on{" "}
                      {data.metadata.refit_samples.toLocaleString()} pre-test
                      origins, still purging before test.
                    </span>
                  </li>
                  <li>
                    <strong>Inspect limitations</strong>
                    <span>
                      Evaluate paired daily blocks, non-overlapping origins,
                      feature ablation and representative errors.
                    </span>
                  </li>
                </ol>
                <SplitTimeline data={data} />
              </section>
              <div className="two-column section">
                <section>
                  <h2>Why random splits would mislead</h2>
                  <p>
                    Neighboring origins share historical inputs and future
                    targets. A shuffled split can put nearly identical intervals
                    into both fitting and evaluation. Chronology better
                    represents forecasting later observations.
                  </p>
                  <p>
                    Historical readings from earlier periods are allowed at
                    validation and test origins: by then they have been
                    observed. Future target readings are never features. This is
                    rolling-origin evaluation with observed feedback, not a
                    month-long open-loop forecast.
                  </p>
                  <p>
                    Imputation and scaling live inside scikit-learn pipelines.
                    Each fitting fold learns its own statistics. Tree models do
                    not need scaling; Ridge does.
                  </p>
                </section>
                <section>
                  <h2>What the metrics mean</h2>
                  <p>
                    <strong>MAE:</strong> average absolute error in Wh per
                    future-hour forecast. Primary selection metric, easy to
                    interpret physically.
                  </p>
                  <p>
                    <strong>RMSE:</strong> square-root mean squared error in Wh.
                    Larger mistakes receive more weight.
                  </p>
                  <p>
                    <strong>R²:</strong> improvement in squared error over the
                    evaluation-set target mean; negative values are possible.
                  </p>
                  <p>
                    <strong>Relative MAE reduction:</strong> 100 × (persistence
                    MAE − model MAE) / persistence MAE. Normalized MAE uses test
                    mean energy as its denominator. MAPE is omitted because
                    small loads make percentages unstable.
                  </p>
                </section>
              </div>
              <section className="section">
                <h2>Feature inventory</h2>
                <p>
                  Calendar features wrap cyclically: sine and cosine keep 23:50
                  close to 00:00. Environmental inputs are household sensors
                  only. Every listed offset is relative to origin t.
                </p>
                <details>
                  <summary>
                    Inspect all {data.features.length} feature definitions
                  </summary>
                  <div
                    className="table-scroll"
                    tabIndex={0}
                    role="region"
                    aria-label="Scrollable research table"
                  >
                    <table>
                      <thead>
                        <tr>
                          <th>Feature</th>
                          <th>Definition</th>
                          <th>Earliest / latest (min)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.features.map((f) => (
                          <tr key={f.name}>
                            <th scope="row">{f.name}</th>
                            <td>{f.definition}</td>
                            <td>
                              {f.earliest_offset_minutes} /{" "}
                              {f.latest_offset_minutes}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
                <a
                  className="download-link"
                  href={BASE + "research/feature_inventory.json"}
                  download
                >
                  <Download size={16} />
                  Download feature audit
                </a>
              </section>
              <section className="section limitations">
                <h2>What this study can’t establish</h2>
                <p>
                  One home and roughly four and a half months cannot establish
                  generalization to other households or future seasons. Occupant
                  behavior can change. Sensor availability is assumed, not
                  measured in a deployed system.
                </p>
                <p>
                  The source supplies no timezone offsets or sensor-release
                  latency. Interpolated airport weather was excluded, but
                  household averaging still depends on the stated
                  end-of-interval availability assumption.
                </p>
                <p>
                  We forecast aggregate appliance energy, without device
                  disaggregation, live control, tariff savings or carbon claims.
                  Overlapping targets make the apparent sample count larger than
                  the amount of independent evidence.
                </p>
                <p>
                  Conclusion: {data.metadata.selected_ml}{" "}
                  {primary.improvement_pct > 0
                    ? "outperformed"
                    : "did not outperform"}{" "}
                  persistence in this held-out record, with a relative MAE
                  reduction of {n(primary.improvement_pct)}%. The result
                  concerns this household and time period; unpredictable
                  activity remains a material limitation.
                </p>
              </section>
            </>
          ) : null}
          {tab === "reproduce" ? (
            <>
              <section className="page-heading">
                <h1>Follow the evidence.</h1>
                <p>
                  Offline Python experiments. Static, transparent research
                  artifacts.
                </p>
              </section>
              <div className="two-column">
                <section>
                  <h2>Reproduce the investigation</h2>
                  <pre tabIndex={0}>
                    git clone {REPO}.git{"\n"}cd smartenergy-ml{"\n"}python -m
                    venv .venv{"\n"}# Windows PowerShell{"\n"}
                    .\.venv\Scripts\Activate.ps1{"\n"}pip install -r
                    requirements.txt{"\n"}python -m src.run_pipeline --overwrite
                    {"\n"}python -m unittest discover -s tests -v
                  </pre>
                  <p>
                    The repository includes the original evaluated artifacts.
                    The overwrite flag explicitly reproduces the frozen
                    protocol. No GPU or inference service is needed.
                  </p>
                  <h2>Run the website</h2>
                  <pre tabIndex={0}>
                    cd website{"\n"}npm ci{"\n"}npm run dev{"\n"}# Production
                    checks{"\n"}npm run build{"\n"}npm run test:e2e
                  </pre>
                </section>
                <section>
                  <h2>Inspect the artifacts</h2>
                  <div className="download-list">
                    {[
                      [
                        "test_predictions.csv",
                        "All held-out predictions (CSV)",
                      ],
                      ["model_comparison.csv", "Model metrics (CSV)"],
                      [
                        "experiment_metadata.json",
                        "Experiment metadata (JSON)",
                      ],
                      ["feature_inventory.json", "Feature definitions (JSON)"],
                    ].map(([file, label]) => (
                      <a key={file} href={BASE + "research/" + file} download>
                        <Download size={17} />
                        {label}
                      </a>
                    ))}
                  </div>
                  <p>
                    Executed: {data.metadata.executed_utc}. Seed{" "}
                    {data.metadata.seed}. Runtime{" "}
                    {n(data.metadata.elapsed_seconds, 0)} seconds on the
                    recorded CPU environment.
                  </p>
                  <pre tabIndex={0}>
                    {JSON.stringify(data.metadata.versions, null, 2)}
                  </pre>
                  <p className="hash">
                    Dataset CSV SHA-256: {data.metadata.dataset.sha256}
                  </p>
                  <a href={REPO} target="_blank" rel="noreferrer">
                    Read the source and viva notes <ExternalLink size={14} />
                  </a>
                </section>
              </div>
              <section className="section">
                <h2>Sources & credits</h2>
                <p>
                  Candanedo, L. (2017).{" "}
                  <a
                    href="https://doi.org/10.24432/C5VC8G"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Appliances Energy Prediction, UCI Machine Learning
                    Repository
                  </a>
                  . CC BY 4.0.
                </p>
                <p>
                  Candanedo, Feldheim & Deramaix (2017).{" "}
                  <a
                    href="https://doi.org/10.1016/j.enbuild.2017.01.083"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Data driven prediction models of energy use of appliances in
                    a low-energy house
                  </a>
                  . Energy and Buildings 140, 81–97.
                </p>
                <p>
                  <a
                    href="https://scikit-learn.org/stable/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    scikit-learn
                  </a>{" "}
                  for classical ML; NumPy, Pandas, SciPy and Matplotlib for
                  analysis. React, Vite, Recharts, Lucide and Manrope for the
                  interface.{" "}
                  <a
                    href="https://ml2026.floodresq.com/project-guide"
                    target="_blank"
                    rel="noreferrer"
                  >
                    ES 335 project guide
                  </a>
                  .
                </p>
                <h2>AI assistance & authorship</h2>
                <p>
                  Codex assisted with methodology implementation, coding,
                  debugging, experiment execution, visual design and drafting.
                  An AI-generated interface concept guided layout; every chart
                  and research number comes from executed Python. The student
                  must review the outputs and understand and defend the choices.
                </p>
                <p>
                  Soumil Shah (24110354) · ES 335 Machine Learning · Individual
                  project.{" "}
                  <a
                    href={REPO + "/blob/main/docs/AI_DISCLOSURE.md"}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Read the full AI disclosure
                  </a>
                  .
                </p>
              </section>
            </>
          ) : null}
          <footer>
            SmartEnergy · Historical evidence, with its limits in view.
            <a href={BASE + "research/experiment_metadata.json"}>
              Experiment record
            </a>
          </footer>
        </main>
      </div>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
