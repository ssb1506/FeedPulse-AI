"""
Iceberg Store — Writes enriched posts to Apache Iceberg lakehouse tables.
Uses PyIceberg with a local SQLite catalog and file-based storage.
"""

import os
import pyarrow as pa
from datetime import datetime, timezone
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    StringType,
    FloatType,
    IntegerType,
    TimestamptzType,
    NestedField,
)
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import HourTransform
import threading


# Schema for our social media posts table
POSTS_SCHEMA = Schema(
    NestedField(1, "id", StringType(), required=True),
    NestedField(2, "username", StringType(), required=True),
    NestedField(3, "text", StringType(), required=True),
    NestedField(4, "topic", StringType(), required=True),
    NestedField(5, "timestamp", TimestamptzType(), required=True),
    NestedField(6, "likes", IntegerType(), required=True),
    NestedField(7, "reposts", IntegerType(), required=True),
    NestedField(8, "sentiment_label", StringType(), required=True),
    NestedField(9, "sentiment_score", FloatType(), required=True),
    NestedField(10, "keywords", StringType(), required=False),  # comma-separated
)

# Partition by hour — enables efficient time-travel queries
POSTS_PARTITION = PartitionSpec(
    PartitionField(source_id=5, field_id=1001, transform=HourTransform(), name="timestamp_hour")
)

WAREHOUSE_PATH = "/tmp/iceberg_warehouse"
CATALOG_DB = f"{WAREHOUSE_PATH}/catalog.db"


class IcebergStore:
    """Manages Iceberg table for social media posts."""

    def __init__(self):
        os.makedirs(WAREHOUSE_PATH, exist_ok=True)

        self.catalog = SqlCatalog(
            "feedpulse",
            **{
                "uri": f"sqlite:///{CATALOG_DB}",
                "warehouse": f"file://{WAREHOUSE_PATH}",
            },
        )

        self._ensure_table()
        self._buffer: list[dict] = []
        self._buffer_lock = threading.Lock()
        self._flush_size = 10  # flush every 10 posts

    def _ensure_table(self):
        """Create the posts table if it doesn't exist."""
        try:
            # Create namespace
            try:
                self.catalog.create_namespace("social")
            except Exception:
                pass  # namespace might already exist

            # Create table
            try:
                self.table = self.catalog.create_table(
                    identifier="social.posts",
                    schema=POSTS_SCHEMA,
                    partition_spec=POSTS_PARTITION,
                )
                print("[Iceberg] Created table 'social.posts'")
            except Exception:
                self.table = self.catalog.load_table("social.posts")
                print("[Iceberg] Loaded existing table 'social.posts'")

        except Exception as e:
            print(f"[Iceberg] Error setting up table: {e}")
            raise

    def append_post(self, post: dict):
        """Buffer a post and flush when buffer is full."""
        with self._buffer_lock:
            self._buffer.append(post)
            if len(self._buffer) >= self._flush_size:
                self._flush()

    def _flush(self):
        """Write buffered posts to Iceberg as a Parquet batch."""
        if not self._buffer:
            return

        try:
            rows = []
            for post in self._buffer:
                rows.append({
                    "id": post["id"],
                    "username": post["username"],
                    "text": post["text"],
                    "topic": post["topic"],
                    "timestamp": datetime.fromisoformat(post["timestamp"]),
                    "likes": post.get("likes", 0),
                    "reposts": post.get("reposts", 0),
                    "sentiment_label": post.get("sentiment_label", "neutral"),
                    "sentiment_score": float(post.get("sentiment_score", 0.0)),
                    "keywords": ",".join(post.get("keywords", [])),
                })

            # Create PyArrow table
            # NOTE: nullable=False on every field except "keywords" — this must
            # match the Iceberg table schema (POSTS_SCHEMA above), which marks
            # these same fields as `required`. PyArrow fields default to
            # nullable=True, which silently fails Iceberg's schema validation
            # on every append if left unset.
            arrow_table = pa.Table.from_pylist(
                rows,
                schema=pa.schema([
                    pa.field("id", pa.string(), nullable=False),
                    pa.field("username", pa.string(), nullable=False),
                    pa.field("text", pa.string(), nullable=False),
                    pa.field("topic", pa.string(), nullable=False),
                    pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
                    pa.field("likes", pa.int32(), nullable=False),
                    pa.field("reposts", pa.int32(), nullable=False),
                    pa.field("sentiment_label", pa.string(), nullable=False),
                    pa.field("sentiment_score", pa.float32(), nullable=False),
                    pa.field("keywords", pa.string(), nullable=True),
                ]),
            )

            # Append to Iceberg table
            self.table.append(arrow_table)
            print(f"[Iceberg] Flushed {len(self._buffer)} posts to lakehouse")
            self._buffer.clear()

        except Exception as e:
            print(f"[Iceberg] Error flushing: {e}")

    def query_recent(self, hours: int = 1) -> list[dict]:
        """Query posts from the last N hours using Iceberg scan."""
        try:
            scan = self.table.scan()
            df = scan.to_pandas()

            if df.empty:
                return []

            # Filter by time
            cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=hours)
            df = df[df["timestamp"] >= cutoff]

            return df.to_dict(orient="records")
        except Exception as e:
            print(f"[Iceberg] Query error: {e}")
            return []

    def query_by_topic(self, topic: str) -> list[dict]:
        """Query posts filtered by topic."""
        try:
            scan = self.table.scan()
            df = scan.to_pandas()

            if df.empty:
                return []

            df = df[df["topic"] == topic]
            return df.tail(50).to_dict(orient="records")
        except Exception as e:
            print(f"[Iceberg] Query error: {e}")
            return []

    def get_sentiment_over_time(self, hours: int = 6) -> list[dict]:
        """Get sentiment averages grouped by hour — time-travel analytics."""
        try:
            scan = self.table.scan()
            df = scan.to_pandas()

            if df.empty:
                return []

            cutoff = datetime.now(timezone.utc) - __import__("datetime").timedelta(hours=hours)
            df = df[df["timestamp"] >= cutoff]

            df["hour"] = df["timestamp"].dt.floor("h")
            grouped = df.groupby("hour").agg(
                avg_sentiment=("sentiment_score", "mean"),
                post_count=("id", "count"),
                positive=("sentiment_label", lambda x: (x == "positive").sum()),
                negative=("sentiment_label", lambda x: (x == "negative").sum()),
                neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
            ).reset_index()

            grouped["hour"] = grouped["hour"].astype(str)
            return grouped.to_dict(orient="records")

        except Exception as e:
            print(f"[Iceberg] Time query error: {e}")
            return []

    def get_snapshot_count(self) -> int:
        """Get number of Iceberg snapshots — shows time-travel capability."""
        try:
            return len(list(self.table.metadata.snapshots))
        except Exception:
            return 0
