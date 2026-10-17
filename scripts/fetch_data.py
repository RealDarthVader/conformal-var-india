"""CLI script to ingest, clean, validate, and cache 15+ years of Indian equity data."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conformal_var.data.loader import NSEDataLoader


def setup_logging() -> None:
    """Configure structured logging for data ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Execute data fetching, cleaning, and caching pipeline."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Ingest, clean, align, and cache Indian equity daily data into Parquet files."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2010-01-01",
        help="Start date in YYYY-MM-DD format (default: 2010-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-10-01",
        help="End date in YYYY-MM-DD format (default: 2026-10-01)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached Parquet files exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/data",
        help="Output directory for cached Parquet files (default: artifacts/data)",
    )
    args = parser.parse_args()

    logging.info("Starting Indian Equity Data Ingestion Pipeline...")
    logging.info(
        "Date Range: %s to %s | Force reload: %s",
        args.start_date,
        args.end_date,
        args.force,
    )

    loader = NSEDataLoader(
        start_date=args.start_date,
        end_date=args.end_date,
    )

    try:
        returns_df, ohlcv_df = loader.run_pipeline(
            force_reload=args.force,
            artifacts_dir=args.output_dir,
        )
    except Exception as exc:
        logging.exception("Failed to execute data pipeline: %s", exc)
        return 1

    out_dir = Path(args.output_dir)
    returns_file = out_dir / "returns.parquet"
    ohlcv_file = out_dir / "ohlcv.parquet"

    logging.info("Data pipeline successfully finished!")
    logging.info("NSE Trading Days Ingested: %d", len(returns_df))
    logging.info("Universe Size: %d tickers", len(returns_df.columns))
    logging.info(
        "Returns Date Range: %s to %s",
        returns_df.index[0].date(),
        returns_df.index[-1].date(),
    )
    logging.info("Returns Parquet Size: %.2f KB", returns_file.stat().st_size / 1024)
    logging.info("OHLCV Parquet Size: %.2f KB", ohlcv_file.stat().st_size / 1024)

    return 0


if __name__ == "__main__":
    sys.exit(main())
