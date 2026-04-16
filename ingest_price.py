"""
ingest_prices.py
----------------
Pulls daily OHLCV price data for all 11 S&P 500 sector ETFs from Yahoo Finance
via the yfinance library and inserts one document per ticker-date into the
`prices` collection in MongoDB Atlas.

Key design decisions:
- Uses upsert (update_one with upsert=True) so the script is safe to re-run
  without creating duplicate documents. If a document for a given ticker-date
  already exists, it gets updated rather than duplicated.
- Computes pct_change and direction (the target variable) at ingestion time
  so they are always available in the database without needing a pipeline step.
- Logs every major operation to prices_ingest.log so you can audit what
  happened if something goes wrong.

Usage:
    python ingest_prices.py

Requirements:
    pip install yfinance pymongo python-dotenv
    MongoDB Atlas connection string set in a .env file as MONGO_URI
"""

import logging
import os
from datetime import datetime

import yfinance as yf
import pandas as pd
from pymongo import MongoClient, ASCENDING
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv


# Configuration


# All 11 SPDR sector ETFs covering the full S&P 500
TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE", "XLB", "XLC"]

# Five years of data — covers COVID crash, rate-hike cycle, and AI bull market
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"

DB_NAME         = "stock_forecast"
COLLECTION_NAME = "prices"

# Logging setup — writes to both the console and a log file


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("prices_ingest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# MongoDB connection

def get_mongo_collection():
    """
    Connects to MongoDB Atlas using the MONGO_URI environment variable.
    Creates a unique compound index on (ticker, date) so no duplicate
    ticker-date pairs can exist in the collection.

    Returns:
        pymongo Collection object for the prices collection.
    """
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError("MONGO_URI not found in environment. Add it to your .env file.")

    client = MongoClient(uri)
    db     = client[DB_NAME]
    col    = db[COLLECTION_NAME]

    # Unique compound index — enforces one document per ticker per date
    # background=True means the index builds without blocking other operations
    col.create_index(
        [("ticker", ASCENDING), ("date", ASCENDING)],
        unique=True,
        background=True
    )
    logger.info("Connected to MongoDB Atlas — collection: %s", COLLECTION_NAME)
    return col


# Data fetching and transformation

def fetch_ticker_data(ticker: str) -> pd.DataFrame:
    """
    Downloads daily OHLCV data for a single ticker from Yahoo Finance.
    Computes pct_change and the binary direction label (next-day target).

    Args:
        ticker: ETF ticker symbol string, e.g. "XLK"

    Returns:
        DataFrame with columns: date, open, high, low, close, volume,
        pct_change, direction — one row per trading day.
    """
    logger.info("Fetching %s from Yahoo Finance (%s to %s)", ticker, START_DATE, END_DATE)

    # auto_adjust=True gives split-and-dividend-adjusted close prices,
    # which is what you want for any multi-year analysis
    raw = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if raw.empty:
        logger.warning("No data returned for %s — skipping", ticker)
        return pd.DataFrame()

    # yfinance returns a MultiIndex column when downloading a single ticker
    # Flatten it if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.reset_index()

    # Convert date to ISO 8601 string — consistent format for MongoDB storage
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # pct_change: day-over-day return as a decimal (e.g. 0.0126 = +1.26%)
    df["pct_change"] = df["close"].pct_change().round(6)

    # direction: 1 if price went up, 0 if flat or down
    # This is the classification target for the ML model
    df["direction"] = (df["pct_change"] > 0).astype(int)

    # Drop the first row — it has no valid pct_change since there's no prior day
    df = df.dropna(subset=["pct_change"])

    # Round price fields to 4 decimal places for clean storage
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].round(4)

    df["ticker"] = ticker
    df["volume"]  = df["volume"].astype(int)

    logger.info("  -> %d trading days fetched for %s", len(df), ticker)
    return df


def build_document(row: pd.Series) -> dict:
    """
    Converts a DataFrame row into a MongoDB document dict.
    All field names are lowercase and snake_case for consistency.

    Args:
        row: A pandas Series representing one trading day for one ticker.

    Returns:
        Dictionary ready for MongoDB insertion.
    """
    return {
        "ticker":     row["ticker"],
        "date":       row["date"],
        "open":       float(row["open"]),
        "high":       float(row["high"]),
        "low":        float(row["low"]),
        "close":      float(row["close"]),
        "volume":     int(row["volume"]),
        "pct_change": float(row["pct_change"]),
        "direction":  int(row["direction"]),
        "ingested_at": datetime.utcnow().isoformat()  # audit timestamp
    }



# Insertion logic

def upsert_ticker(col, ticker: str) -> int:
    """
    Fetches data for one ticker and upserts all documents into MongoDB.
    Using upsert means re-running the script is always safe — it won't
    create duplicates, it will just update existing documents.

    Args:
        col:    pymongo Collection object
        ticker: ETF ticker symbol string

    Returns:
        Number of documents successfully upserted.
    """
    df = fetch_ticker_data(ticker)
    if df.empty:
        return 0

    upserted = 0
    for _, row in df.iterrows():
        doc = build_document(row)
        try:
            # filter_key uniquely identifies the document — ticker + date
            filter_key = {"ticker": doc["ticker"], "date": doc["date"]}
            col.update_one(filter_key, {"$set": doc}, upsert=True)
            upserted += 1
        except Exception as e:
            logger.error("Failed to upsert %s on %s: %s", ticker, doc["date"], e)

    logger.info("  -> Upserted %d documents for %s", upserted, ticker)
    return upserted


# Main entry point

def main():
    logger.info("=" * 60)
    logger.info("Starting price data ingestion")
    logger.info("Tickers: %s", TICKERS)
    logger.info("Date range: %s to %s", START_DATE, END_DATE)
    logger.info("=" * 60)

    try:
        col = get_mongo_collection()
    except EnvironmentError as e:
        logger.critical("Cannot connect to MongoDB: %s", e)
        return

    total = 0
    for ticker in TICKERS:
        try:
            n = upsert_ticker(col, ticker)
            total += n
        except Exception as e:
            logger.error("Unexpected error processing %s: %s", ticker, e)

    logger.info("=" * 60)
    logger.info("Ingestion complete. Total documents upserted: %d", total)
    logger.info("Expected: ~%d (11 tickers × ~1,260 trading days)", 11 * 1260)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
