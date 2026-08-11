"""Metric history, trends, and per-host thresholds.

The point of this module is one question a snapshot cannot answer: **is this normal?**

"Memory is at 74%" is not information. Memory at 74% on a host that has sat at 72% for a month
is fine. Memory at 74% on a host that was at 40% on Monday is a leak, and it will page someone
at 3am on Saturday. Same reading, opposite meaning, and the difference is entirely history.

Everything here is deterministic arithmetic over stored numbers. No model is involved in
deciding whether something is trending — the LLM's job is to explain a trend, never to detect
one by eyeballing a list of readings.
"""

from datetime import datetime, timedelta, timezone

from db import get_conn, tx, utcnow

# Enough history to see a working week, at a resolution that survives a year of polling
# without anyone thinking about it. SQLite handles this table size without complaint; the
# retention exists so nobody discovers a 4 GB database in eight months.
RETENTION_DAYS = 90

DEFAULT_WARN_PCT = 85.0
DEFAULT_CRIT_PCT = 95.0

# Below this many samples, "the trend is up" is noise dressed as a finding.
MIN_SAMPLES_FOR_TREND = 5

# A metric has to move by more than this across the window before it is called a direction
# rather than jitter. CPU in particular wanders several points between any two reads.
TREND_SIGNIFICANCE_PCT = 5.0


# --- writing -------------------------------------------------------------------------------

def record_snapshot(org_id: str, host_id: str, snap: dict) -> None:
    """Extract the numeric spine of a snapshot into the time series.

    Silently tolerant of missing probes: a host where `connections` needs elevation still has
    perfectly good CPU and memory readings, and refusing to store those because one probe
    was unavailable would lose the history that matters most."""
    def num(*path, default=None):
        node = snap
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key)
        return node if isinstance(node, (int, float)) else default

    with tx() as c:
        c.execute(
            "INSERT INTO metrics (host_id, org_id, recorded_at, cpu, memory, disk, swap, "
            "process_count, thread_count, uptime_seconds) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (host_id, org_id, utcnow(),
             num("cpu", "percent"), num("memory", "percent"), num("disk", "percent"),
             num("swap", "percent"), num("process_count", "processes"),
             num("process_count", "threads"), num("uptime", "uptime_seconds")),
        )


def prune(days: int = RETENTION_DAYS) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with tx() as c:
        cur = c.execute("DELETE FROM metrics WHERE recorded_at < ?", (cutoff,))
    return cur.rowcount


# --- reading -------------------------------------------------------------------------------

_METRIC_COLUMNS = ("cpu", "memory", "disk", "swap", "process_count",
                   "thread_count", "uptime_seconds")


def series(host_id: str, metric: str, hours: int = 24, limit: int = 500) -> list[dict]:
    if metric not in _METRIC_COLUMNS:
        raise ValueError(f"Unknown metric '{metric}'.")
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = get_conn()
    rows = conn.execute(
        f"SELECT recorded_at, {metric} AS value FROM metrics "
        f"WHERE host_id = ? AND recorded_at >= ? AND {metric} IS NOT NULL "
        f"ORDER BY recorded_at DESC LIMIT ?",
        (host_id, since, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank. With the sample counts here, interpolating between neighbours would be
    false precision on top of readings that are already ±1%."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered)) - 1))
    return ordered[idx]


def trend(host_id: str, metric: str, hours: int = 24) -> dict:
    """Baseline statistics plus a direction, for one metric over one window.

    `direction` compares the first third of the window against the last third rather than
    first-sample-to-last: two readings can differ by ten points from ordinary jitter, and a
    trend built on the endpoints alone reports a rise every time the last sample happens to
    land high."""
    points = series(host_id, metric, hours=hours)
    values = [p["value"] for p in points if p["value"] is not None]
    if len(values) < MIN_SAMPLES_FOR_TREND:
        return {"metric": metric, "window_hours": hours, "samples": len(values),
                "sufficient_data": False,
                "note": f"Need at least {MIN_SAMPLES_FOR_TREND} readings to describe a trend."}

    # series() returns newest-first; reverse so "first third" means oldest.
    chrono = list(reversed(values))
    third = max(1, len(chrono) // 3)
    early = sum(chrono[:third]) / third
    late = sum(chrono[-third:]) / third
    delta = late - early

    if abs(delta) < TREND_SIGNIFICANCE_PCT:
        direction = "stable"
    elif delta > 0:
        direction = "rising"
    else:
        direction = "falling"

    return {
        "metric": metric,
        "window_hours": hours,
        "samples": len(values),
        "sufficient_data": True,
        "current": chrono[-1],
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "avg": round(sum(values) / len(values), 1),
        "p95": round(_percentile(values, 95), 1),
        "baseline_early": round(early, 1),
        "baseline_late": round(late, 1),
        "change": round(delta, 1),
        "direction": direction,
    }


def summary(host_id: str, hours: int = 24) -> dict:
    """Trends for every metric worth trending, in one call."""
    return {m: trend(host_id, m, hours) for m in ("cpu", "memory", "disk", "swap")}


def anomalies(host_id: str, hours: int = 24) -> list[dict]:
    """Findings a single reading cannot produce.

    Disk is called out separately from CPU and memory: CPU spiking and settling is a Tuesday,
    but disk only ever goes one way, so a rising disk trend is a date with an outage that can
    be calculated in advance."""
    found = []
    for metric in ("cpu", "memory", "disk", "swap"):
        t = trend(host_id, metric, hours)
        if not t.get("sufficient_data"):
            continue
        if t["direction"] == "rising":
            level = "WARNING" if metric in ("memory", "disk") else "INFO"
            found.append({
                "metric": metric, "level": level, "kind": "rising_trend",
                # ASCII only. A '->' arrow rather than '→' because these strings reach Windows
                # consoles and log files, where the default cp1252 codec raises
                # UnicodeEncodeError on the arrow and takes the whole write down with it.
                "detail": (f"{metric} has risen {t['change']}% over {hours}h "
                           f"({t['baseline_early']}% -> {t['baseline_late']}%)."),
            })
        if t["current"] > t["p95"] and t["samples"] >= 20:
            found.append({
                "metric": metric, "level": "INFO", "kind": "above_baseline",
                "detail": (f"{metric} is at {t['current']}%, above its own 95th percentile "
                           f"of {t['p95']}% for the last {hours}h."),
            })
    return found


# --- per-host thresholds ---------------------------------------------------------------------

def get_thresholds(host_id: str) -> dict:
    conn = get_conn()
    row = conn.execute("SELECT warn_pct, crit_pct FROM host_settings WHERE host_id = ?",
                       (host_id,)).fetchone()
    warn = row["warn_pct"] if row and row["warn_pct"] is not None else DEFAULT_WARN_PCT
    crit = row["crit_pct"] if row and row["crit_pct"] is not None else DEFAULT_CRIT_PCT
    return {"warn": float(warn), "crit": float(crit),
            "customised": bool(row and (row["warn_pct"] is not None
                                        or row["crit_pct"] is not None))}


def set_thresholds(host_id: str, warn: float | None, crit: float | None,
                   notes: str | None = None) -> dict:
    if warn is not None and crit is not None and warn >= crit:
        raise ValueError("The warning threshold must be below the critical threshold.")
    for value in (warn, crit):
        if value is not None and not (0 < float(value) <= 100):
            raise ValueError("Thresholds must be a percentage between 0 and 100.")
    with tx() as c:
        c.execute(
            "INSERT INTO host_settings (host_id, warn_pct, crit_pct, notes, updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(host_id) DO UPDATE SET "
            "warn_pct=excluded.warn_pct, crit_pct=excluded.crit_pct, "
            "notes=excluded.notes, updated_at=excluded.updated_at",
            (host_id, warn, crit, notes, utcnow()),
        )
    return get_thresholds(host_id)
