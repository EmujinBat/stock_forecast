"""
ingest_macro.py
---------------
Pulls six macroeconomic time series from the Federal Reserve Economic Database
(FRED) via the fredapi Python library and inserts one document per trading day
into the `macro_indicators` collection in MongoDB Atlas.

The six series collected are:
  - DFF       : Effective federal funds rate (daily)
  - DGS10     : 10-year Treasury constant maturity yield (daily)
  - VIXCLS    : CBOE VIX volatility index (daily)
  - CPIAUCSL  : Consumer Price Index, all urban consumers (monthly)
  - UNRATE    : US unemployment rate (monthly)
  - UMCSENT   : University of Michigan Consumer Sentiment Index (monthly)

Monthly series are forward-filled to align with the daily trading calendar.
This reflects what a market participant would actually have known on any given
trading day — the most recent published value — which is the right approach
for a forecasting exercise.

Usage:
    python ingest_macro.py

Requirements:
    pip install fredapi pymongo pandas python-dotenv
    FRED_API_KEY set in .env (free key from fred.stlouisfed.org)
    MONGO_URI set in .env
"""

import logging
import os
from datetime import datetime

import pandas as pd
from fredapi import Fred
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# FRED series IDs mapped to human-readable field names
FRED_SERIES = {
    "DFF":      "fed_funds_rate",     # daily
    "DGS10":    "treasury_10yr",      # daily
    "VIXCLS":   "vix",                # daily
    "CPIAUCSL": "cpi_yoy",            # monthly — we compute YoY change below
    "UNRATE":   "unemployment_rate",  # monthly
    "UMCSENT":  "consumer_sentiment", # monthly
}

START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

DB_NAME         = "stock_forecast"
COLLECTION_NAME = "macro_indicators"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("macro_ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MongoDB connection
# ---------------------------------------------------------------------------

def get_mongo_collection():
    """
    Connects to MongoDB Atlas and returns the macro_indicators collection.
    Creates a unique index on `date` so there is exactly one macro document
    per trading day.

    Returns:
        pymongo Collection object
    """
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI not found in environment.")

    client = MongoClient(uri)
    col    = client[DB_NAME][COLLECTION_NAME]

    # One document per trading day — date must be unique
    col.create_index([("date", ASCENDING)], unique=True, background=True)
    logger.info("Connected to MongoDB Atlas — collection: %s", COLLECTION_NAME)
    return col


# ---------------------------------------------------------------------------
# FRED data fetching
# ---------------------------------------------------------------------------

def fetch_all_series() -> pd.DataFrame:
    """
    Fetches all six FRED series and merges them into a single daily DataFrame
    aligned to the trading calendar.

    CPI is converted to year-over-year percent change (the more interpretable
    form) rather than the raw index level. All monthly series are forward-filled
    to populate daily values between releases.

    Returns:
        DataFrame indexed by date with one column per macro indicator,
        restricted to trading days between START_DATE and END_DATE.
    """
    load_dotenv()
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not found in environment. "
                               "Get a free key at fred.stlouisfed.org/docs/api/api_key.html")

    fred = Fred(api_key=api_key)

    # Build a daily date range covering our full window
    daily_index = pd.date_range(start=START_DATE, end=END_DATE, freq="B")  # B = business days
    df = pd.DataFrame(index=daily_index)
    df.index.name = "date"

    for series_id, field_name in FRED_SERIES.items():
        logger.info("Fetching FRED series: %s -> %s", series_id, field_name)
        try:
            raw = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)

            if series_id == "CPIAUCSL":
                # Convert raw CPI index to year-over-year percent change
                # This is the form markets actually watch and discuss
                raw = raw.pct_change(periods=12) * 100
                raw = raw.round(3)

            # Reindex to our daily date range, then forward-fill
            # Forward-fill is the correct approach — it represents what
            # was publicly known on each trading day
            reindexed = raw.reindex(daily_index, method=None)
            filled     = reindexed.ffill()

            # Track which dates were filled (not natively daily)
            df[field_name]            = filled.round(4)
            df[f"_{field_name}_filled"] = reindexed.isna()  # True = this day was forward-filled

            logger.info("  -> %d values fetched, %d days forward-filled",
                        raw.notna().sum(), reindexed.isna().sum())

        except Exception as e:
            logger.error("Failed to fetch %s: %s", series_id, e)
            df[field_name] = None

    return df


# ---------------------------------------------------------------------------
# Insertion logic
# ---------------------------------------------------------------------------

def build_fill_notes(row: pd.Series) -> str:
    """
    Constructs a human-readable string noting which fields were forward-filled
    on this particular date. This is stored in the document for transparency.

    Args:
        row: A DataFrame row for one date.

    Returns:
        String like "cpi_yoy filled; unemployment_rate filled" or "" if none.
    """
    filled_fields = [
        field for field in FRED_SERIES.values()
        if row.get(f"_{field}_filled", False)
    ]
    return "; ".join(f"{f} forward-filled" for f in filled_fields)


def upsert_all(col, df: pd.DataFrame) -> int:
    """
    Iterates over all trading days and upserts one macro document per day.

    Args:
        col: pymongo Collection object
        df:  DataFrame with macro indicator columns, indexed by date

    Returns:
        Total number of documents successfully upserted.
    """
    upserted = 0
    for date, row in df.iterrows():
        date_str   = date.strftime("%Y-%m-%d")
        fill_notes = build_fill_notes(row)

        doc = {
            "date":               date_str,
            "fed_funds_rate":     float(row["fed_funds_rate"])     if pd.notna(row.get("fed_funds_rate"))     else None,
            "treasury_10yr":      float(row["treasury_10yr"])      if pd.notna(row.get("treasury_10yr"))      else None,
            "vix":                float(row["vix"])                if pd.notna(row.get("vix"))                else None,
            "cpi_yoy":            float(row["cpi_yoy"])            if pd.notna(row.get("cpi_yoy"))            else None,
            "unemployment_rate":  float(row["unemployment_rate"])  if pd.notna(row.get("unemployment_rate"))  else None,
            "consumer_sentiment": float(row["consumer_sentiment"]) if pd.notna(row.get("consumer_sentiment")) else None,
            "ingested_at":        datetime.utcnow().isoformat(),
        }

        # Only add _fill_notes if there's actually something to note
        if fill_notes:
            doc["_fill_notes"] = fill_notes

        try:
            col.update_one({"date": date_str}, {"$set": doc}, upsert=True)
            upserted += 1
        except Exception as e:
            logger.error("Failed to upsert macro doc for %s: %s", date_str, e)

    return upserted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Starting macro indicator ingestion")
    logger.info("Series: %s", list(FRED_SERIES.keys()))
    logger.info("Date range: %s to %s", START_DATE, END_DATE)
    logger.info("=" * 60)

    try:
        col = get_mongo_collection()
    except EnvironmentError as e:
        logger.critical("Cannot connect to MongoDB: %s", e)
        return

    try:
        df = fetch_all_series()
    except EnvironmentError as e:
        logger.critical("Cannot connect to FRED: %s", e)
        return

    total = upsert_all(col, df)

    logger.info("=" * 60)
    logger.info("Macro ingestion complete. Documents upserted: %d", total)
    logger.info("Expected: ~1,260 (trading days in 2020-2024)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()