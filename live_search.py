# live_search.py
"""
On-demand SGCarMart used-car listing fetcher (requests + BeautifulSoup only).

- Fetches ONE page (up to 60 cars) from:
    https://www.sgcarmart.com/used-cars/listing
- Applies optional filters: budget_min, budget_max, make
- Extracts key fields using robust regex on each listing block
- Uses the stable pattern div[id^="listing_"] to capture each car card
- Returns a pandas DataFrame for use by Streamlit / value_model.py
"""

import re
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

# --------- HTTP headers ---------
UAS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
]

BASE_URL = "https://www.sgcarmart.com/used-cars/listing"


def _headers() -> dict:
    """Return a random realistic desktop User-Agent header."""
    return {"User-Agent": random.choice(UAS)}


# --------- small helpers ---------
def _clean(t: str | None) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", t).strip()


def _to_int(s: str | None) -> int | None:
    if not s:
        return None
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s.isdigit() else None


# --------- main function ---------
def fetch_used_cars_live(
    budget_min: int | None = None,
    budget_max: int | None = None,
    make: str | None = None,
    max_results: int = 40,
) -> pd.DataFrame:
    """
    Fetch one page of SGCarMart used-car listings.

    Parameters
    ----------
    budget_min : int | None
        Minimum price in SGD.
    budget_max : int | None
        Maximum price in SGD.
    make : str | None
        Car brand, e.g. "Toyota", "Honda".
    max_results : int
        Maximum number of cars to return (<= 60).

    Returns
    -------
    pandas.DataFrame
        Columns: title, listing_url, price_sgd, mileage_km,
                 depreciation_per_year, year, coe_left_years,
                 engine_cc, raw_text
    """

    params: dict[str, str | int] = {
        "avl": "a",   # available cars
        "limit": 60,  # 60 per page (max)
        "page": 1,
    }

    if budget_min:
        params["PR1"] = budget_min
    if budget_max:
        params["PR2"] = budget_max
    if make:
        params["Make"] = make

    print("\n[SGCM] GET:", BASE_URL)
    print("[SGCM] Params:", params)

    try:
        resp = requests.get(BASE_URL, params=params, headers=_headers(), timeout=20)
    except Exception as e:
        print("[ERROR] Network error:", e)
        return pd.DataFrame()

    if resp.status_code != 200:
        print(f"[ERROR] Status {resp.status_code} from SGCM")
        return pd.DataFrame()

    html = resp.text
    print("[DEBUG] HTML length:", len(html))

    soup = BeautifulSoup(html, "html.parser")

    # ----- the key: each listing row has id="listing_0", "listing_1", ... -----
    cards = soup.select('div[class^="styles_listing_box"]')
    print(f"[DEBUG] Found {len(cards)} vehicle cards.")

    if not cards:
        print("[ERROR] No car listing blocks found. Layout may have changed or request blocked.")
        return pd.DataFrame()

    rows: list[dict] = []

    for card in cards:
        # full card text (safe)
        raw_text = _clean(card.get_text(" ") or "")

        # ---------- title & URL ----------
        link_el = card.select_one("a[href*='/used-cars/info/']")
        if link_el:
            title = _clean(link_el.get_text())
            href = link_el.get("href") or ""
            if href.startswith("/"):
                listing_url = "https://www.sgcarmart.com" + href
            else:
                listing_url = href or None
        else:
            title = None
            listing_url = None

        # ---------- price ----------
        price_sgd = None
        if raw_text:
            mp = re.search(r"\$\s*([\d,]+)", raw_text)
            if mp:
                price_sgd = _to_int(mp.group(1))

        # ---------- mileage ----------
        mileage_km = None
        if raw_text:
            mm = re.search(r"([\d,]+)\s*km\b", raw_text, re.I)
            if mm:
                mileage_km = _to_int(mm.group(1))

        # ---------- depreciation per year ----------
        depreciation_per_year = None
        if raw_text:
            md = re.search(r"Depreciation[^$]*\$\s*([\d,]+)/yr", raw_text, re.I)
            if not md:
                md = re.search(r"\$\s*([\d,]+)/yr", raw_text, re.I)
            if md:
                depreciation_per_year = _to_int(md.group(1))

        # ---------- engine capacity (cc) ----------
        engine_cc = None
        if raw_text:
            me = re.search(r"(\d{3,4})\s*cc\b", raw_text, re.I)
            if me:
                engine_cc = _to_int(me.group(1))

        # ---------- year (registration or model year) ----------
        year = None
        if raw_text:
            # pick the first 4-digit year starting with 19/20
            my = re.search(r"\b(20\d{2}|19\d{2})\b", raw_text)
            if my:
                try:
                    year = int(my.group(1))
                except ValueError:
                    year = None

        # ---------- COE left (years) ----------
        coe_left_years = None
        if raw_text:
            mc = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:yr|yrs)\s*COE\s*left",
                raw_text,
                re.I,
            )
            if mc:
                try:
                    coe_left_years = float(mc.group(1))
                except ValueError:
                    coe_left_years = None

        rows.append(
            {
                "title": title,
                "listing_url": listing_url,
                "price_sgd": price_sgd,
                "mileage_km": mileage_km,
                "depreciation_per_year": depreciation_per_year,
                "year": year,
                "coe_left_years": coe_left_years,
                "engine_cc": engine_cc,
                "raw_text": raw_text,
            }
        )

        if len(rows) >= max_results:
            break

    df = pd.DataFrame(rows)
    print(f"[SGCM] Returning {len(df)} cars")
    return df
