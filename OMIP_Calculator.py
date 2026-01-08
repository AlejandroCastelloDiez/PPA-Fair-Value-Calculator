#!/usr/bin/env python3
"""
Daily OMIP -> PPA FV (ES/PT) updater.

Outputs (repo root):
- PPA prices ES PT.json   (append-only by date)
- PPA_FV_ES.png
- PPA_FV_PT.png
- PPA_ES_PT_Basis.png
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from calendar import isleap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# =========================
# Config
# =========================
REPO_ROOT = Path(__file__).resolve().parent

PPA_JSON_NAME = "PPA prices ES PT.json"
PPA_JSON_PATH = REPO_ROOT / PPA_JSON_NAME

# Image names requested (no dates)
IMG_ES = REPO_ROOT / "PPA_FV_ES.png"
IMG_PT = REPO_ROOT / "PPA_FV_PT.png"
IMG_BASIS = REPO_ROOT / "PPA_ES_PT_Basis.png"

TENORS = [5, 7, 10]  # years
DEFAULT_DISCOUNT_RATE = 0.05
USE_LEAP_HOURS = True

TIMEZONE = "Europe/Madrid"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

OMIP_URL = "https://www.omip.pt/es/dados-mercado"  # params: date, instrument, product, zone


# =========================
# Styling (robust font fallback)
# =========================
def _set_roboto_font() -> None:
    """
    Tries to use Roboto if available; falls back to DejaVu Sans.
    Never hard-fails if font isn't installed on runner.
    """
    try:
        candidates = [f for f in fm.findSystemFonts(fontext="ttf") if "Roboto" in Path(f).name]
        if candidates:
            roboto_path = sorted(candidates)[0]
            fm.fontManager.addfont(roboto_path)
            roboto_name = fm.FontProperties(fname=roboto_path).get_name()
            plt.rcParams["font.family"] = roboto_name
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        plt.rcParams["font.family"] = "DejaVu Sans"


# =========================
# Market specs
# =========================
@dataclass(frozen=True)
class MarketSpec:
    zone: str        # "ES" / "PT"
    instrument: str  # "FTB" / "FPB"
    product: str     # "EL"


MARKETS: Dict[str, MarketSpec] = {
    "ES": MarketSpec(zone="ES", instrument="FTB", product="EL"),
    "PT": MarketSpec(zone="PT", instrument="FPB", product="EL"),
}


# =========================
# Helpers
# =========================
def yyyymmdd_to_iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


def iso_to_yyyymmdd(iso: str) -> str:
    return iso.replace("-", "")


def get_yesterday_yyyymmdd(tz_name: str = TIMEZONE) -> str:
    if ZoneInfo is None:
        # fallback: naive local time (still fine for GitHub runner most days)
        d = (datetime.now() - timedelta(days=1)).date()
    else:
        now_tz = datetime.now(ZoneInfo(tz_name))
        d = (now_tz - timedelta(days=1)).date()
    return d.strftime("%Y%m%d")


def hours_in_year(year: int, use_leap_hours: bool = True) -> int:
    if not use_leap_hours:
        return 8760
    return 8784 if isleap(year) else 8760


def _safe_float(x: str) -> Optional[float]:
    s = (x or "").strip()
    if not s or s.lower() in {"n.a.", "na", "n/a", "-"}:
        return None
    # OMIP uses dot decimals; keep robust to commas anyway
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


# =========================
# OMIP scraping (Year reference prices)
# =========================
def fetch_omip_year_reference_prices(
    asof_iso_date: str,
    market: MarketSpec,
    session: Optional[requests.Session] = None,
) -> Dict[int, float]:
    """
    Returns mapping delivery_year -> D reference price (€/MWh) for Year contracts
    (e.g., YR-27 -> 2027) for the given market and date.

    Scrapes https://www.omip.pt/es/dados-mercado?date=YYYY-MM-DD&instrument=...&product=EL&zone=...
    and parses table "Reference prices" (column 'D (€/MWh)').
    """
    sess = session or requests.Session()
    params = {
        "date": asof_iso_date,
        "instrument": market.instrument,
        "product": market.product,
        "zone": market.zone,
    }
    headers = {"User-Agent": USER_AGENT}
    r = sess.get(OMIP_URL, params=params, headers=headers, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Find the table that contains the header "Contract name" and "D (€/MWh)"
    tables = soup.find_all("table")
    target = None
    header_cells: List[str] = []
    for t in tables:
        thead = t.find("thead")
        if not thead:
            continue
        ths = [th.get_text(" ", strip=True) for th in thead.find_all("th")]
        if any("Contract name" in h for h in ths) and any(re.search(r"\bD\b", h) for h in ths):
            target = t
            header_cells = ths
            break

    if target is None:
        # As a fallback, try the first table with many rows
        if tables:
            target = max(tables, key=lambda x: len(x.find_all("tr")))
            header_cells = [th.get_text(" ", strip=True) for th in target.find_all("th")]

    if target is None:
        raise RuntimeError("Could not locate OMIP reference prices table in HTML.")

    # Identify column indexes
    # Typical headers include: "Contract name", ... , "D (€/MWh)", "D-1 (€/MWh)" ...
    def find_col(pattern: str) -> Optional[int]:
        for i, h in enumerate(header_cells):
            if re.search(pattern, h, flags=re.IGNORECASE):
                return i
        return None

    col_contract = find_col(r"Contract\s*name") or 0
    col_d = find_col(r"^D\b")  # header like "D (€/MWh)"
    if col_d is None:
        # try contains "D (€/MWh)"
        col_d = find_col(r"\bD\s*\(")

    if col_d is None:
        raise RuntimeError(f"Could not find 'D (€/MWh)' column. Headers: {header_cells}")

    out: Dict[int, float] = {}

    tbody = target.find("tbody") or target
    rows = tbody.find_all("tr")
    for tr in rows:
        tds = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if not tds or len(tds) <= max(col_contract, col_d):
            continue

        contract = tds[col_contract]
        # We only want Year contracts like "FTB YR-27" / "FPB YR-28"
        m = re.search(r"\bYR-(\d{2})\b", contract)
        if not m:
            continue

        yy = int(m.group(1))
        # interpret YY as 2000+YY (safe for OMIP)
        delivery_year = 2000 + yy

        d_val = _safe_float(tds[col_d])
        if d_val is None:
            continue

        out[delivery_year] = d_val

    if not out:
        raise RuntimeError(
            f"No Year reference prices found for {market.zone}/{market.instrument} on {asof_iso_date}."
        )

    return out


# =========================
# PPA FV calculation
# =========================
def ppa_fv_from_year_forwards(
    asof: date,
    year_forwards: Dict[int, float],
    tenor_years: int,
    *,
    discount_rate: float,
    use_leap_hours: bool,
) -> float:
    """
    Converts yearly forward prices into a single fixed PPA price for a flat 1 MW baseload,
    using mid-year discounting and hour weighting.

    Start year = asof.year + 1
    Term covers start_year ... start_year+tenor_years-1
    Discount time uses mid-year: (year - asof_year) - 0.5
    """
    start_year = asof.year + 1
    years = list(range(start_year, start_year + tenor_years))

    # PV(price * hours) / PV(hours) => levelized fixed price
    pv_num = 0.0
    pv_den = 0.0

    for y in years:
        if y not in year_forwards:
            raise KeyError(f"Missing OMIP YR price for delivery year {y}")

        price = float(year_forwards[y])
        h = float(hours_in_year(y, use_leap_hours=use_leap_hours))

        # Mid-year timing relative to asof year:
        # e.g. asof 2026 -> year 2027 mid-year is ~1.5 years away
        t = (y - asof.year) - 0.5
        df = 1.0 / ((1.0 + discount_rate) ** t) if discount_rate != 0 else 1.0

        pv_num += price * h * df
        pv_den += h * df

    return pv_num / pv_den


# =========================
# JSON store (append-only by date)
# =========================
def load_store(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("JSON store must be an object.")
        if "data" not in obj or not isinstance(obj["data"], dict):
            # normalize
            obj["data"] = {}
        if "meta" not in obj or not isinstance(obj["meta"], dict):
            obj["meta"] = {}
        return obj

    return {"meta": {}, "data": {}}


def save_store(path: Path, obj: Dict[str, Any], pretty: bool = True) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=False)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def update_store_for_date(
    yyyymmdd: str,
    *,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    use_leap_hours: bool = USE_LEAP_HOURS,
    tenors: List[int] = TENORS,
) -> Dict[str, Any]:
    """
    Fetch OMIP data for the given date and append PPA FV values to store if date not present.
    Returns the updated store object.
    """
    iso = yyyymmdd_to_iso(yyyymmdd)
    asof_dt = datetime.strptime(iso, "%Y-%m-%d").date()

    store = load_store(PPA_JSON_PATH)

    # meta refresh (keep prior fields if present)
    store["meta"].setdefault("format", "data[date] = {PPA_ES:{5Y,7Y,10Y}, PPA_PT:{5Y,7Y,10Y}}")
    store["meta"]["discount_rate"] = discount_rate
    store["meta"]["use_leap_hours"] = bool(use_leap_hours)
    store["meta"]["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # append-only
    if iso in store["data"]:
        return store

    with requests.Session() as sess:
        es_years = fetch_omip_year_reference_prices(iso, MARKETS["ES"], session=sess)
        pt_years = fetch_omip_year_reference_prices(iso, MARKETS["PT"], session=sess)

    entry: Dict[str, Any] = {"PPA_ES": {}, "PPA_PT": {}}

    for n in tenors:
        entry["PPA_ES"][f"{n}Y"] = round(
            ppa_fv_from_year_forwards(
                asof_dt, es_years, n, discount_rate=discount_rate, use_leap_hours=use_leap_hours
            ),
            2,
        )
        entry["PPA_PT"][f"{n}Y"] = round(
            ppa_fv_from_year_forwards(
                asof_dt, pt_years, n, discount_rate=discount_rate, use_leap_hours=use_leap_hours
            ),
            2,
        )

    store["data"][iso] = entry
    save_store(PPA_JSON_PATH, store, pretty=True)
    return store


# =========================
# Plotting
# =========================
LINE_COLORS = {"10Y": "#69DECE", "7Y": "#219B93", "5Y": "#1C1C1C"}  # only used if present


def store_to_timeseries_df(store: Dict[str, Any]) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with columns:
    ES_5Y, ES_7Y, ES_10Y, PT_5Y, PT_7Y, PT_10Y
    """
    rows = []
    data = store.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("store['data'] must be a dict.")

    for iso, v in data.items():
        if not isinstance(v, dict):
            continue
        es = (v.get("PPA_ES") or {}) if isinstance(v.get("PPA_ES"), dict) else {}
        pt = (v.get("PPA_PT") or {}) if isinstance(v.get("PPA_PT"), dict) else {}
        rows.append(
            {
                "date": iso,
                "ES_5Y": es.get("5Y"),
                "ES_7Y": es.get("7Y"),
                "ES_10Y": es.get("10Y"),
                "PT_5Y": pt.get("5Y"),
                "PT_7Y": pt.get("7Y"),
                "PT_10Y": pt.get("10Y"),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def plot_zone(df: pd.DataFrame, zone_prefix: str, out_path: Path, title: str) -> None:
    """
    zone_prefix: "ES" or "PT"
    """
    _set_roboto_font()

    cols = [f"{zone_prefix}_5Y", f"{zone_prefix}_7Y", f"{zone_prefix}_10Y"]
    data = df[cols].dropna(how="all")
    if data.empty:
        raise RuntimeError(f"No data available to plot for zone {zone_prefix}.")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for c in cols:
        label = c.split("_")[1]
        series = data[c].dropna()
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=label, linewidth=2.2, color=LINE_COLORS.get(label))

    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("€/MWh")
    ax.grid(False)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_basis(df: pd.DataFrame, out_path: Path) -> None:
    _set_roboto_font()

    basis = pd.DataFrame(index=df.index)
    basis["Premium_5Y"] = df["PT_5Y"] - df["ES_5Y"]
    basis["Premium_7Y"] = df["PT_7Y"] - df["ES_7Y"]
    basis["Premium_10Y"] = df["PT_10Y"] - df["ES_10Y"]
    basis = basis.dropna(how="all")
    if basis.empty:
        raise RuntimeError("No data available to plot basis.")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for col in ["Premium_5Y", "Premium_7Y", "Premium_10Y"]:
        label = col.split("_")[1]
        series = basis[col].dropna()
        if series.empty:
            continue
        ax.plot(series.index, series.values, label=label, linewidth=2.2, color=LINE_COLORS.get(label))

    ax.set_title("PPA Premium (Portugal - Spain) by Tenor", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("€/MWh")
    ax.grid(False)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main() -> None:
    yyyymmdd = get_yesterday_yyyymmdd(TIMEZONE)

    # 1) Update JSON (append-only for yesterday)
    store = update_store_for_date(
        yyyymmdd,
        discount_rate=DEFAULT_DISCOUNT_RATE,
        use_leap_hours=USE_LEAP_HOURS,
        tenors=TENORS,
    )

    # 2) Produce charts from entire stored history
    df = store_to_timeseries_df(store)
    if df.empty:
        raise RuntimeError("Store has no data after update; cannot plot.")

    plot_zone(df, "ES", IMG_ES, "PPA FV — Spain (ES) by Tenor")
    plot_zone(df, "PT", IMG_PT, "PPA FV — Portugal (PT) by Tenor")
    plot_basis(df, IMG_BASIS)

    print(f"Updated: {PPA_JSON_PATH.name} (date added if missing: {yyyymmdd_to_iso(yyyymmdd)})")
    print(f"Saved: {IMG_ES.name}, {IMG_PT.name}, {IMG_BASIS.name}")


if __name__ == "__main__":
    main()

