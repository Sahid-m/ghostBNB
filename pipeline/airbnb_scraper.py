"""
Defensive, best-effort scraper for a SINGLE live Airbnb listing.

Airbnb actively blocks scrapers and changes its page layout often, so this module
is built to NEVER raise out of ``fetch_listing``. It always returns a status dict
and degrades gracefully; the dashboard falls back to manual entry on failure.

Extracted values are mapped onto the live-scrapeable column names declared in
``pipeline/features.py`` so the same vocabulary flows through training/inference.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pprint import pprint

import requests
from bs4 import BeautifulSoup

try:
    from pipeline.features import NUMERIC, CATEGORICAL, BOOL, parse_price, to_bool
except Exception:  # pragma: no cover - allow running as a loose script
    from features import NUMERIC, CATEGORICAL, BOOL, parse_price, to_bool

# Columns we might plausibly read off a live page. (Availability/host-portfolio
# counts and ltm reviews are generally NOT on the public page, so we don't try.)
_KNOWN_COLUMNS = set(NUMERIC) | set(CATEGORICAL) | set(BOOL)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Candidate key-names (any case/separator) -> our feature column. The first
# non-empty hit found while walking the embedded JSON wins.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "room_type": ("roomtype", "room_type", "roomtypecategory", "listingroomtype"),
    "property_type": ("propertytype", "property_type", "roomtypecategory_localized"),
    "accommodates": ("persdoncapacity", "personcapacity", "accommodates", "guestcapacity", "capacity"),
    "bedrooms": ("bedrooms", "bedroomcount", "numberofbedrooms"),
    "beds": ("beds", "bedcount", "numberofbeds"),
    "bathrooms": ("bathrooms", "bathroomcount", "numberofbathrooms", "baths"),
    "minimum_nights": ("minimumnights", "minnights", "minimum_nights", "mininglos"),
    "maximum_nights": ("maximumnights", "maxnights", "maximum_nights"),
    "number_of_reviews": ("reviewscount", "visiblereviewcount", "reviewcount",
                          "number_of_reviews", "numberofreviews", "reviewsavailable"),
    "reviews_per_month": ("reviewspermonth", "reviews_per_month"),
    "host_is_superhost": ("issuperhost", "superhost", "host_is_superhost"),
    "instant_bookable": ("instantbook", "instantbookable", "instant_bookable",
                         "caninstantbook"),
}

# Numeric vs categorical vs bool handling for coercion.
_NUMERIC_COLS = set(NUMERIC)
_BOOL_COLS = set(BOOL)

_ROOM_TYPE_MAP = {
    "entire_home": "Entire home/apt",
    "entire home/apt": "Entire home/apt",
    "entire home": "Entire home/apt",
    "entire_place": "Entire home/apt",
    "entire": "Entire home/apt",
    "private_room": "Private room",
    "private room": "Private room",
    "private": "Private room",
    "shared_room": "Shared room",
    "shared room": "Shared room",
    "shared": "Shared room",
    "hotel_room": "Hotel room",
    "hotel room": "Hotel room",
    "hotel": "Hotel room",
}


def _canonical_room_type(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    key = s.lower().replace("-", "_")
    if key in _ROOM_TYPE_MAP:
        return _ROOM_TYPE_MAP[key]
    # e.g. "Entire rental unit" type strings — return as-is, don't guess.
    return s


def parse_listing_id(id_or_url: str) -> str | None:
    """Bare digits, or any URL containing /rooms/<digits> (query stripped)."""
    if id_or_url is None:
        return None
    s = str(id_or_url).strip()
    if not s:
        return None
    if s.isdigit():
        return s
    m = re.search(r"/rooms/(?:plus/)?(\d+)", s)
    if m:
        return m.group(1)
    # last resort: a long-ish standalone number anywhere in the string
    m = re.search(r"\b(\d{4,})\b", s.split("?")[0])
    return m.group(1) if m else None


def _coerce(col: str, value):
    """Coerce a raw value into the right type for the given feature column."""
    if value is None:
        return None
    if col == "room_type":
        return _canonical_room_type(value)
    if col == "property_type":
        s = str(value).strip()
        return s or None
    if col in _BOOL_COLS:
        b = to_bool(value)
        return None if b != b else bool(b)  # NaN check
    if col in _NUMERIC_COLS:
        num = parse_price(value)
        return None if num != num else num   # NaN check
    s = str(value).strip()
    return s or None


def _norm_key(k) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _walk(obj, found: dict):
    """Recursively collect first-seen values for our candidate key-names."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (str, int, float, bool)) and v is not None:
                nk = _norm_key(k)
                for col, aliases in _KEY_ALIASES.items():
                    if col in found:
                        continue
                    if nk in aliases:
                        coerced = _coerce(col, v)
                        if coerced is not None:
                            found[col] = coerced
            _walk(v, found)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, found)


def _extract_price_from_json(obj):
    """Best-effort price hunt: look for amount-ish keys with a plausible value."""
    result = {"value": None}

    def rec(o):
        if result["value"] is not None:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                nk = _norm_key(k)
                if nk in ("amount", "priceamount", "priceperning", "rate") and isinstance(v, (int, float)):
                    result["value"] = float(v)
                    return
                if nk in ("price", "pricestring", "displayprice", "discountedprice") and isinstance(v, str):
                    p = parse_price(v)
                    if p == p:
                        result["value"] = p
                        return
                rec(v)
        elif isinstance(o, list):
            for item in o:
                rec(item)

    rec(obj)
    return result["value"]


def _blank_result(listing_id, url):
    return {
        "ok": False,
        "listing_id": listing_id,
        "url": url,
        "features": {},
        "raw": {"title": None, "price_text": None, "image": None,
                "rating": None, "reviews_count": None},
        "reason": "",
    }


def fetch_listing(id_or_url: str, timeout: int = 15) -> dict:
    """Best-effort fetch of a single live Airbnb listing. Never raises."""
    listing_id = None
    url = ""
    try:
        listing_id = parse_listing_id(id_or_url)
        if not listing_id:
            res = _blank_result(None, "")
            res["reason"] = "could not parse listing id"
            return res

        url = f"https://www.airbnb.com/rooms/{listing_id}"
        result = _blank_result(listing_id, url)

        # --- fetch with one retry -------------------------------------------
        resp = None
        last_err = None
        for attempt in range(2):
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=timeout,
                                    allow_redirects=True)
                break
            except Exception as e:  # network / timeout
                last_err = e
                if attempt == 0:
                    time.sleep(1.5)
        if resp is None:
            result["reason"] = f"request failed: {last_err}"
            return result

        if resp.status_code == 404:
            result["reason"] = "404 not found"
            return result
        if resp.status_code != 200:
            blocked = " (blocked)" if resp.status_code in (403, 429, 503) else ""
            result["reason"] = f"non-200 response: HTTP {resp.status_code}{blocked}"
            return result

        html = resp.text or ""
        soup = BeautifulSoup(html, "lxml")
        raw = result["raw"]
        features: dict = {}

        # --- (a) og:/product: meta tags -------------------------------------
        def meta(*names):
            for n in names:
                tag = soup.find("meta", attrs={"property": n}) or \
                      soup.find("meta", attrs={"name": n})
                if tag and tag.get("content"):
                    return tag["content"].strip()
            return None

        raw["title"] = meta("og:title", "twitter:title")
        raw["image"] = meta("og:image", "twitter:image")
        price_meta = meta("product:price:amount", "og:price:amount", "og:price")
        if price_meta:
            raw["price_text"] = price_meta
            p = parse_price(price_meta)
            if p == p:
                features.setdefault("price", p)

        # --- (b) embedded JSON ----------------------------------------------
        json_blobs = []
        for sc in soup.find_all("script"):
            sid = (sc.get("id") or "")
            stype = (sc.get("type") or "")
            if ("data-deferred-state" in sid or "data-injector-instances" in sid
                    or stype == "application/json"):
                txt = sc.string or sc.get_text() or ""
                txt = txt.strip()
                if not txt:
                    continue
                try:
                    json_blobs.append(json.loads(txt))
                except Exception:
                    continue

        for blob in json_blobs:
            _walk(blob, features)
        if "price" not in features:
            for blob in json_blobs:
                p = _extract_price_from_json(blob)
                if p is not None:
                    features["price"] = p
                    break

        # --- (c) JSON-LD -----------------------------------------------------
        for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
            txt = (sc.string or sc.get_text() or "").strip()
            if not txt:
                continue
            try:
                ld = json.loads(txt)
            except Exception:
                continue
            items = ld if isinstance(ld, list) else [ld]
            for item in items:
                if not isinstance(item, dict):
                    continue
                if not raw["title"] and item.get("name"):
                    raw["title"] = str(item["name"]).strip()
                if not raw["image"] and item.get("image"):
                    img = item["image"]
                    raw["image"] = img[0] if isinstance(img, list) and img else (
                        img if isinstance(img, str) else raw["image"])
                agg = item.get("aggregateRating")
                if isinstance(agg, dict):
                    if raw["rating"] is None and agg.get("ratingValue") is not None:
                        try:
                            raw["rating"] = float(agg["ratingValue"])
                        except Exception:
                            pass
                    if raw["reviews_count"] is None and agg.get("reviewCount") is not None:
                        try:
                            rc = int(float(agg["reviewCount"]))
                            raw["reviews_count"] = rc
                            features.setdefault("number_of_reviews", float(rc))
                        except Exception:
                            pass

        # --- fill raw rating/reviews from features if not yet set ------------
        if raw["reviews_count"] is None and "number_of_reviews" in features:
            try:
                raw["reviews_count"] = int(features["number_of_reviews"])
            except Exception:
                pass

        # keep only known feature columns, drop Nones
        result["features"] = {k: v for k, v in features.items()
                              if k in _KNOWN_COLUMNS and v is not None}

        # ok if we got a title OR a price
        got_title = bool(raw["title"])
        got_price = "price" in result["features"]
        result["ok"] = got_title or got_price
        if result["ok"]:
            result["reason"] = "ok"
        else:
            result["reason"] = "no embedded data found (title/price not parseable)"
        return result

    except Exception as e:  # absolute last-resort safety net
        res = _blank_result(listing_id, url)
        res["reason"] = f"parse error: {e}"
        return res


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "12345"
    print(f"=== fetch_listing({arg!r}) ===")
    out = fetch_listing(arg)
    pprint(out, sort_dicts=False, width=100)
