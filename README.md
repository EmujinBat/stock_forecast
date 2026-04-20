# DS 4320 Project 2: Forecasting S&P 500 Sector ETF Price Direction Using Price and Macroeconomic Signals

### Executive Summary

This project predicts whether each of the eleven S&P 500 sector ETFs will close higher or lower the following trading day. Two real data sources are combined: daily price data from Yahoo Finance and economic indicators from the Federal Reserve (FRED). All data is stored in MongoDB Atlas across two collections totaling over 15,000 documents. A Random Forest classifier is trained on the merged dataset and evaluated on a held-out time period (2023-2024). Feature importance analysis identifies which signal type (price-based technical indicators or macroeconomic indicators) contributes the most predictive power.

---

### Name: Emujin Batzorig
### NetID: kfm8nx
### DOI: *to be added after Zenodo release*
### Press Release: [Can a Machine Tell You Whether the Stock Market Will Go Up Tomorrow?](./press_release.md)
### Pipeline: [pipeline.ipynb](./pipeline.ipynb) | [pipeline.md](./pipeline.md)
### License: [MIT](./LICENSE)

---

| Spec | Value |
|---|---|
| Name | Emujin Batzorig |
| NetID | kfm8nx |
| DOI | *to be added* |
| Press Release | [Can a Machine Tell You Whether the Stock Market Will Go Up Tomorrow?](./press_release.md) |
| Data | MongoDB Atlas (credentials submitted privately via Canvas) |
| Pipeline | [pipeline.ipynb](./pipeline.ipynb) · [pipeline.md](./pipeline.md) |
| License | [MIT](./LICENSE) |

---

## Problem Definition

### General and Specific Problem

- **General:** Forecast stock prices.
- **Specific:** Predict whether each of eleven S&P 500 sector ETFs will close higher or lower the following trading day, using past price patterns and economic indicators. Then identify which of the two data sources contributes the most to making accurate predictions.

### Motivation

The S&P 500 is made up of eleven sectors (technology, energy, healthcare, etc.) that react differently to the same events, so predicting the market as one single thing misses a lot of useful detail. The core question this project asks is whether combining past price patterns with broader economic conditions produces better predictions than using price data alone. By measuring how much each source contributes to the model's accuracy, the project gives a concrete answer for the 2020-2024 period across all eleven sectors. A document database like MongoDB is a natural fit for storing this data because each source has a different structure and update frequency, and MongoDB lets each collection keep its own format until they are merged at analysis time.

### Rationale for Refinement

Predicting the exact closing price of a stock is not realistic because prices are shaped by global events no model can fully anticipate, so reframing it as a yes/no question (will it go up or down tomorrow?) makes the problem both more honest and easier to evaluate. Individual stocks were ruled out because they are heavily influenced by company-specific events like earnings surprises or leadership changes that are nearly impossible to predict from public data alone. Sector ETFs each hold dozens of stocks, which smooths out that company-level noise and leaves a pattern more likely to be explained by broader economic signals. The eleven SPDR sector ETFs together cover the entire S&P 500 without overlap.

### Press Release

[Can a Machine Tell You Whether the Stock Market Will Go Up Tomorrow?](./press_release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|---|---|
| ETF | A fund that holds many stocks and trades like a single stock. Sector ETFs hold stocks from just one industry group. |
| OHLCV | Open, High, Low, Close, Volume: the five standard fields recorded for each trading day. |
| Directional movement | Whether the closing price went up (1) or down (0) compared to the day before. This is the target variable the model predicts. |
| RSI | Relative Strength Index: a score from 0 to 100 measuring how fast a stock has been moving. Above 70 may signal it is overpriced; below 30 may signal it is undervalued. |
| Moving average | The average closing price over a recent window (e.g., the past 5 or 20 days). Used to detect short-term price trends. |
| FRED | Federal Reserve Economic Data: a free public database of US economic statistics maintained by the St. Louis Federal Reserve. |
| VIX | A number that measures how nervous investors are about the stock market. High VIX means high uncertainty. Sometimes called the "fear index." |
| Federal funds rate | The interest rate the Federal Reserve sets for overnight bank lending. When it rises, borrowing becomes more expensive across the whole economy. |
| CPI | Consumer Price Index: tracks how much everyday goods and services cost over time. The main measure of inflation. |
| Treasury yield (10yr) | The interest rate on a 10-year US government bond. A useful signal for how investors feel about long-term economic growth. |
| Random Forest | A machine learning model that builds many decision trees and combines their predictions. Known for being accurate and interpretable. |
| Feature importance | A score the Random Forest produces for each input variable, showing how useful that variable was for making predictions. |
| F1-score | A combined measure of prediction accuracy that accounts for both false positives and false negatives. |
| Soft schema | A set of rules for how documents should be structured, enforced by the code rather than by the database itself. |

### Domain Overview

This project combines stock market data with machine learning. The core idea is that stock prices do not move randomly: past price patterns and economic conditions both carry some information about where prices might go next. The goal is not to predict the future perfectly (that is not possible) but to find out whether combining two different types of data produces more accurate predictions than using either one alone.

Stock market prediction is a well-studied problem in both academia and industry. One consistent finding in the research literature is that past returns do carry a weak short-term signal: stocks that went up recently tend to continue going up slightly. Economic indicators like interest rates and inflation shape the environment that all stocks operate in, and different sectors respond to those conditions in very different ways.

### Background Reading: [link to folder](https://drive.google.com/drive/folders/1gFMeaUus-46b0RDuQ7ND1TfUOT8oy4f8?usp=sharing)

| Title | Description | Link |
|---|---|---|
| Jegadeesh & Titman (1993): Returns to Buying Winners and Selling Losers | Shows that recent past performance predicts near-future returns, which is the academic justification for using lagged returns as model features. | [PDF](https://drive.google.com/file/d/1MJIl9EK7TxtSXcDG20JHDWkmQ3AK1V9C/view?usp=sharing) |
| Fama (1970): Efficient Capital Markets: A Review of Theory and Evidence | The classic argument that stock prices already reflect all available information, making them impossible to predict consistently. This project tests that assumption. | [PDF](https://drive.google.com/file/d/1UZHWxMBV2YaxxDABwRVUdawhOOlASnHv/view?usp=sharing) |
| Sharpe (1964): Capital Asset Prices: A Theory of Market Equilibrium | Foundational paper on how risk and expected return relate across different assets and sectors. Provides context for why different sectors behave differently under the same economic conditions. | [PDF](https://drive.google.com/file/d/1lakFwbkXbrfIGkFUXdI_m59MDuyZN8_p/view?usp=sharing) |
| Ang & Bekaert (2007): Stock Return Predictability: Is it There? | Reviews the empirical evidence on whether stock returns can be predicted using financial and economic variables, directly relevant to the forecasting task in this project. | [PDF](https://drive.google.com/file/d/1FUGp57kunddD56RXlqo3TVf1goKtplS4/view?usp=sharing) |
| Campbell & Shiller (1988): Stock Prices, Earnings, and Expected Dividends | Examines the relationship between macroeconomic fundamentals and stock price movements, supporting the inclusion of economic indicators as predictive features. | [PDF](https://drive.google.com/file/d/1heV9DzK6WohR6F887ebb-aKCln7SbKcL/view?usp=sharing) |

---

## Data Creation

### Provenance

All data is real and publicly available. The two sources were accessed programmatically using Python libraries and loaded directly into MongoDB Atlas via the ingestion scripts in this repository.

**Source 1: Yahoo Finance via yfinance**
Price data was collected using the `yfinance` Python library, which wraps Yahoo Finance's public data endpoint. No API key is required. The library's `yf.download()` function was called once per ticker with `auto_adjust=True` to retrieve split- and dividend-adjusted prices, and the raw response was transformed into per-day documents before insertion. The full acquisition process is documented in `ingest_prices.py`.

**Source 2: FRED API via fredapi**
Economic indicator data was collected using the `fredapi` Python library, which connects to the Federal Reserve's official public API. A free API key is required and can be obtained at fred.stlouisfed.org. Each series was fetched individually using `fred.get_series()` with a fixed date range, then reindexed to the daily trading calendar and forward-filled to account for series that are only published monthly. The full acquisition process, including how forward-filling is applied and tracked, is documented in `ingest_macro.py`.

### Code Table

| File | Description | Link |
|---|---|---|
| `ingest_prices.py` | Downloads daily price data for all 11 ETFs from Yahoo Finance and loads it into the `prices` collection. Computes `pct_change` and `direction` at ingestion time. | [ingest_prices.py](./ingest_prices.py) |
| `ingest_macro.py` | Downloads six economic indicator series from FRED, forward-fills to the daily trading calendar, and loads into `macro_indicators`. Records which fields were filled in `_fill_notes`. | [ingest_macro.py](./ingest_macro.py) |
| `pipeline.ipynb` | Full analysis pipeline: queries MongoDB, merges both collections, builds features, trains and evaluates the Random Forest model, and produces visualizations. | [pipeline.ipynb](./pipeline.ipynb) |

### Rationale for Critical Decisions

**Sector ETFs over individual stocks:** Individual company stocks are heavily influenced by events specific to that company (earnings reports, leadership changes, legal issues) that are very hard to predict from public information. ETFs average across many companies, leaving a cleaner signal that broader data can actually explain.

**Five years of data (2020-2024):** Long enough to include four very different market conditions, which forces the model to find patterns that hold across situations rather than memorizing one period. XLC (Communications) launched in 2018, making 2020 a clean common start date for all eleven tickers.

**Forward-filling economic data:** Monthly economic reports represent what was publicly known between release dates. Applying the most recent available value to each day accurately reflects the information a real investor would have had on any given day.

**Time-based train/test split (train: 2020-2022, test: 2023-2024):** With time-series data, randomly splitting into train and test sets causes a problem where the model trains on data from the future relative to some of its test data. Using a strict date cutoff avoids this and gives a realistic picture of how the model would perform going forward.

### Bias Identification

**Survivorship bias:** All eleven SPDR ETFs exist today with continuous data through the full five years. Any sector fund that was closed or restructured during this period would not be included.

**Timing bias in economic data:** Economic reports are published with a delay (the inflation report, for example, arrives about 10 days after the month it covers). If the report's value were applied to the same month it describes rather than to the date it was actually published, the model would be using information that was not yet available. This project aligns indicators to their actual publication dates.

**Price-only signal limitation:** By using only price and macroeconomic data, this project does not capture information that news or earnings announcements might carry. On days with major company or sector news, the model has no way of knowing that something unusual is happening, which may reduce its accuracy on high-news days compared to quieter periods.

### Bias Mitigation

The timing bias in economic data is handled by matching each indicator to its actual publication date rather than its reference period. The five-year window covering four distinct market periods reduces the risk that the model is only learning patterns from one specific situation. Feature importance output is used to check whether any single variable is so dominant that it suggests a data problem rather than genuine predictive power.

---

## Metadata

### Soft-Schema Guidelines

MongoDB does not enforce a fixed structure on documents. The following rules are enforced by the ingestion scripts and should be followed by anyone working with this database.

**Collection: `prices`**
One document per ticker per trading day. Required fields: `ticker` (string, one of the eleven SPDR tickers), `date` (string, YYYY-MM-DD), `open`, `high`, `low`, `close` (floats, adjusted for dividends and stock splits), `volume` (integer), `pct_change` (float), `direction` (integer, 1 = up, 0 = down). The pair (ticker, date) must be unique. Documents without a valid `close` or `direction` are rejected.

**Collection: `macro_indicators`**
One document per trading day. Required fields: `date` (string, YYYY-MM-DD), `fed_funds_rate`, `treasury_10yr`, `vix`, `cpi_yoy`, `unemployment_rate`, `consumer_sentiment` (all floats). Each date must be unique. Any forward-filled values are noted in the optional `_fill_notes` field.

### Data Summary

| Collection | Documents | Date Range | Unique Tickers |
|---|---|---|---|
| `prices` | 13,860 | 2020-01-02 to 2024-12-31 | 11 |
| `macro_indicators` | 1,260 | 2020-01-02 to 2024-12-31 | N/A |
| **Total** | **15,120** | | |

### Data Dictionary

#### prices

| Feature | Type | Description | Example |
|---|---|---|---|
| `ticker` | String | ETF ticker symbol (uppercase). | `"XLK"` |
| `date` | String | Trading date, YYYY-MM-DD. | `"2024-03-15"` |
| `open` | Float | Opening price in USD (adjusted). | `224.10` |
| `high` | Float | Highest price reached during the session (USD). | `227.85` |
| `low` | Float | Lowest price reached during the session (USD). | `223.40` |
| `close` | Float | Closing price in USD (adjusted for dividends and splits). | `226.92` |
| `volume` | Integer | Number of shares traded during the session. | `4812000` |
| `pct_change` | Float | Percent change from the prior day's close, in decimal form. | `0.0126` |
| `direction` | Integer | Target variable: 1 if price went up, 0 if flat or down. | `1` |

#### macro_indicators

| Feature | Type | Description | Example |
|---|---|---|---|
| `date` | String | Trading date, YYYY-MM-DD. | `"2024-03-15"` |
| `fed_funds_rate` | Float | Federal Reserve's overnight lending rate (%). | `5.33` |
| `treasury_10yr` | Float | 10-year US government bond yield (%). | `4.38` |
| `vix` | Float | Market fear index (higher = more uncertainty). | `21.4` |
| `cpi_yoy` | Float | Year-over-year inflation rate (%, forward-filled monthly). | `2.6` |
| `unemployment_rate` | Float | Share of the labor force without a job (%, forward-filled monthly). | `4.1` |
| `consumer_sentiment` | Float | University of Michigan survey measuring consumer confidence (forward-filled monthly). | `79.4` |
| `_fill_notes` | String | Optional. Notes which fields were forward-filled and from what date. | `"cpi_yoy filled from 2024-03-01"` |

### Uncertainty Quantification

| Feature | Type of Uncertainty | Quantification |
|---|---|---|
| `close` | Measurement | Yahoo Finance adjusts historical prices retroactively when dividends or stock splits occur, meaning stored values can change after ingestion. For the SPDR ETFs in this dataset, dividend yields range from roughly 1.3% (XLK) to 3.8% (XLU) annually, representing the upper bound on how much a historical close value might shift. |
| `pct_change` | Derived / propagated | Computed as (close - prev_close) / prev_close. For the eleven ETFs over 2020-2024, daily returns ranged from -12% to +12%, with a mean near 0% and standard deviation of roughly 1.2% per day. The bid-ask spread for liquid ETFs like these is typically under $0.01, introducing a maximum measurement error of under 0.005% per price point. |
| `volume` | Measurement | Reported as regular-session volume only. Across the dataset, daily volume ranges from roughly 500,000 to 80,000,000 shares depending on the ETF and market conditions. Pre- and post-market volume is excluded, which typically accounts for 5-15% of total daily volume on active trading days. |
| `direction` | Label / derived | Binary label derived from `pct_change`. Across all eleven ETFs over 2020-2024, the rate of "up" days is approximately 54% (ranging from 51% for XLE to 57% for XLK). A model that always predicts "up" would achieve 54% accuracy with no actual learning, which is the baseline this project must beat. |
| `fed_funds_rate` | Forward-fill | The rate was changed 11 times between 2020 and 2024, ranging from 0.08% (March 2020) to 5.33% (mid-2023). On the roughly 1,249 days between rate changes, the forward-filled value is exact. On the approximately 11 announcement days, a one-day lag is possible if FRED does not update immediately, introducing a potential error of 0.25 to 0.75 percentage points for one day per change. |
| `cpi_yoy` | Release-lag | Released monthly with a roughly 10-day delay. The forward-filled value can be up to 40 trading days old at the end of each fill window. Over the dataset, CPI ranged from -0.1% to 9.1% year-over-year. The maximum error from using a stale value is bounded by the month-over-month change, which averaged 0.3 percentage points during 2022 (the most volatile inflation period). |
| `vix` | Market / daily | VIX changes continuously throughout the day; the stored value is the daily close. Over the dataset, VIX ranged from 11.5 (very calm) to 85.5 (peak COVID panic in March 2020), with a median of about 19. The intraday range on volatile days can exceed 5 points, meaning the close-of-day snapshot may differ from the intraday average by up to 10-15%. |

---

## License

MIT License: see [LICENSE](./LICENSE)
