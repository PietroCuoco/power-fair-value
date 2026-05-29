"""SMARD.de data ingestion.

SMARD (Bundesnetzagentur / German regulator) publishes German electricity
market data under a CC BY 4.0 licence via an undocumented-but-stable JSON
endpoint. No API token is required.

Endpoint shape (see github.com/bundesAPI/smard-api):

* index   : /chart_data/{filter}/{region}/index_{resolution}.json
            -> {"timestamps": [ms_epoch, ...]}  (weekly block start times, UTC)
* series  : /chart_data/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json
            -> {"meta_data": {...}, "series": [[ms_epoch, value|null], ...]}

Timestamps are millisecond UNIX epoch in UTC. ``null`` values mean "missing"
and are mapped to NaN (handled downstream in QA).

Filter IDs below are taken from the SMARD OpenAPI spec. The day-ahead price is
served for the DE/LU market area (region ``DE-LU``); load and generation series
use region ``DE``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

SMARD_BASE = "https://www.smard.de/app/chart_data"
DEFAULT_RESOLUTION = "hour"
_REQUEST_TIMEOUT = 60
_MAX_RETRIES = 4
_BACKOFF_SECONDS = 1.5


@dataclass(frozen=True)
class Series:
    """A logical SMARD series: a filter id, a region, and a 'kind'.

    ``kind`` is either ``"actual"`` (realised, only known after the fact) or
    ``"forecast"`` (day-ahead forecast, available before gate closure). The kind
    matters for the leakage guard added in the modelling stage: only forecast
    and lagged-actual series are admissible as features for next-day prices.
    """

    filter_id: int
    region: str
    kind: str  # "actual" | "forecast"


# Confirmed against the SMARD OpenAPI spec. The forecast-load id is intentionally
# absent here and confirmed at runtime via ``discover`` (see module docstring).
SERIES_REGISTRY: dict[str, Series] = {
    # Target
    "price_da": Series(4169, "DE-LU", "actual"),  # day-ahead clearing price (€/MWh)
    # Demand side (realised)
    "load_actual": Series(410, "DE", "actual"),
    "residual_load_actual": Series(4359, "DE", "actual"),
    # Renewable supply (realised) - used for the oracle ablation, never as a feature
    "gen_wind_onshore_actual": Series(4067, "DE", "actual"),
    "gen_wind_offshore_actual": Series(1225, "DE", "actual"),
    "gen_pv_actual": Series(4068, "DE", "actual"),
    # Day-ahead forecasts (admissible point-in-time features)
    "fc_gen_total": Series(122, "DE", "forecast"),
    "fc_gen_wind_pv": Series(5097, "DE", "forecast"),  # key renewable driver
    "fc_gen_wind_onshore": Series(123, "DE", "forecast"),
    "fc_gen_wind_offshore": Series(3791, "DE", "forecast"),
    "fc_gen_pv": Series(125, "DE", "forecast"),
    # Day-ahead forecasted total load (Netzlast). Validated by magnitude: mean
    # ~53 GW matches German load. The other live candidate (filter 413) averaged
    # ~0.09 GW, far from the ~34 GW residual-load level reported by the
    # Bundesnetzagentur, so it is NOT residual load and is excluded. The residual
    # load forecast is constructed in the feature layer as
    # fc_load_total - fc_gen_wind_pv.
    "fc_load_total": Series(411, "DE", "forecast"),
}

# Candidate filter ids to probe for a day-ahead *load* forecast. We do not know
# the exact id from the public spec, so ``discover`` checks which respond.
FORECAST_LOAD_CANDIDATES: tuple[int, ...] = (411, 412, 413, 414, 415, 416, 6, 7)


def _get_json(url: str, session: requests.Session | None = None) -> dict:
    """GET a URL and return parsed JSON, with simple exponential backoff."""
    sess = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = sess.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                raise FileNotFoundError(f"404 for {url}")
            resp.raise_for_status()
            return resp.json()
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry on any transient error
            last_exc = exc
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")


# --- Pure parsers (no network; unit-tested directly) -----------------------


def parse_index(payload: dict) -> list[int]:
    """Extract the sorted list of weekly block start timestamps (ms epoch)."""
    return sorted(int(t) for t in payload.get("timestamps", []))


def parse_block(payload: dict) -> pd.Series:
    """Turn a SMARD series block into a UTC-indexed float Series.

    ``null`` values become NaN. The index is a tz-aware UTC DatetimeIndex.
    """
    rows = payload.get("series", [])
    if not rows:
        return pd.Series(dtype="float64")
    ts = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    vals = pd.to_numeric(pd.Series([r[1] for r in rows]), errors="coerce")
    return pd.Series(vals.to_numpy(), index=ts, dtype="float64")


def _select_blocks(block_starts: list[int], start_ms: int, end_ms: int) -> list[int]:
    """Pick the weekly blocks needed to cover [start_ms, end_ms]."""
    in_range = [t for t in block_starts if t <= end_ms]
    earlier = [t for t in in_range if t <= start_ms]
    if earlier:
        cutoff = earlier[-1]  # include the block containing start
        in_range = [t for t in in_range if t >= cutoff]
    return in_range


# --- Network fetchers -------------------------------------------------------


def fetch_series(
    series: Series,
    start: str,
    end: str,
    resolution: str = DEFAULT_RESOLUTION,
    session: requests.Session | None = None,
) -> pd.Series:
    """Fetch a single series over [start, end] as a UTC-indexed Series."""
    f, region = series.filter_id, series.region
    index_url = f"{SMARD_BASE}/{f}/{region}/index_{resolution}.json"
    block_starts = parse_index(_get_json(index_url, session))
    if not block_starts:
        return pd.Series(dtype="float64")

    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    needed = _select_blocks(block_starts, start_ms, end_ms)

    parts: list[pd.Series] = []
    for ts in needed:
        url = f"{SMARD_BASE}/{f}/{region}/{f}_{region}_{resolution}_{ts}.json"
        try:
            parts.append(parse_block(_get_json(url, session)))
        except FileNotFoundError:
            continue  # block listed but not yet materialised

    if not parts:
        return pd.Series(dtype="float64")

    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def build_dataset(cfg: dict, save: bool = True) -> pd.DataFrame:
    """Fetch every registered series and assemble a wide UTC-indexed frame."""
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"]
    if end == "auto":
        end = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

    session = requests.Session()
    frames: dict[str, pd.Series] = {}
    for name, series in SERIES_REGISTRY.items():
        print(f"  fetching {name} (filter {series.filter_id}, {series.region}) ...")
        frames[name] = fetch_series(series, start, end, session=session)

    df = pd.DataFrame(frames).sort_index()
    df.index.name = "timestamp_utc"

    if save:
        out_path = Path(cfg["data"]["raw_dir"]) / "smard_raw.parquet"
        df.to_parquet(out_path)
        print(f"  saved {len(df):,} rows x {df.shape[1]} cols -> {out_path}")
    return df


def discover(
    candidates: tuple[int, ...] = FORECAST_LOAD_CANDIDATES,
    region: str = "DE",
    resolution: str = DEFAULT_RESOLUTION,
) -> pd.DataFrame:
    """Probe candidate filter ids and report which return data and their span.

    Run this once to confirm the day-ahead load-forecast filter id, then add it
    to ``SERIES_REGISTRY`` and ``config.yaml``.
    """
    session = requests.Session()
    rows = []
    for f in candidates:
        url = f"{SMARD_BASE}/{f}/{region}/index_{resolution}.json"
        try:
            ts = parse_index(_get_json(url, session))
            if ts:
                first = pd.to_datetime(ts[0], unit="ms", utc=True).date()
                last = pd.to_datetime(ts[-1], unit="ms", utc=True).date()
                rows.append({"filter": f, "available": True, "from": first, "to": last})
            else:
                rows.append({"filter": f, "available": False, "from": None, "to": None})
        except Exception as exc:  # noqa: BLE001
            rows.append({"filter": f, "available": False, "from": None, "to": str(exc)[:40]})
    return pd.DataFrame(rows)