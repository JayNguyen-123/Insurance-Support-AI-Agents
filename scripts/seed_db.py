#!/usr/bin/env python3
"""CLI: (re)build the SQLite database and the FAQ vector store.

Usage:
    python scripts/seed_db.py               # seed both DB and FAQ collection
    python scripts/seed_db.py --db-only
    python scripts/seed_db.py --faq-only
"""

from __future__ import annotations

import argparse
import sys

from app.db.bootstrap import setup_insurance_database
from app.logging_config import configure_logging, get_logger
from app.vectorstore.faq_store import seed_faq_collection

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-only", action="store_true", help="Only seed the SQLite database")
    parser.add_argument("--faq-only", action="store_true", help="Only seed the FAQ vector store")
    args = parser.parse_args()

    configure_logging()

    do_db = not args.faq_only
    do_faq = not args.db_only

    if do_db:
        logger.info("Seeding SQLite database with synthetic sample data...")
        setup_insurance_database()
        logger.info("Database seeding complete.")

    if do_faq:
        logger.info("Seeding FAQ vector store...")
        count = seed_faq_collection()
        logger.info("FAQ vector store seeding complete.", extra={"documents_indexed": count})

    return 0


if __name__ == "__main__":
    sys.exit(main())
