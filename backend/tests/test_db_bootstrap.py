import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from postgrest.types import ReturnMethod


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.db.bootstrap import chunked_upsert


class FakeUpsertTable:
    def __init__(self) -> None:
        self.calls = []

    def upsert(self, records, **kwargs):
        self.calls.append({"records": records, "kwargs": kwargs})
        return self

    def execute(self):
        return None


class FakeClient:
    def __init__(self) -> None:
        self.tables: dict[str, FakeUpsertTable] = {}

    def table(self, name: str) -> FakeUpsertTable:
        self.tables.setdefault(name, FakeUpsertTable())
        return self.tables[name]


class DbBootstrapTestCase(unittest.TestCase):
    def test_chunked_upsert_defaults_to_returning_minimal(self):
        client = FakeClient()

        chunked_upsert(
            client,
            "price_data",
            [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            on_conflict="ticker,date,source",
        )

        call = client.tables["price_data"].calls[0]
        self.assertEqual(call["kwargs"]["on_conflict"], "ticker,date,source")
        self.assertEqual(call["kwargs"]["returning"], ReturnMethod.minimal)

    def test_chunked_upsert_keeps_returning_override_for_small_dry_runs(self):
        client = FakeClient()

        chunked_upsert(
            client,
            "stock_info",
            [{"ticker": "AAPL"}],
            on_conflict="ticker",
            returning=ReturnMethod.representation,
        )

        call = client.tables["stock_info"].calls[0]
        self.assertEqual(call["kwargs"]["returning"], ReturnMethod.representation)

    def test_chunked_upsert_passes_returning_minimal_to_each_chunk(self):
        client = FakeClient()
        records = [{"id": index} for index in range(5)]

        with patch("backend.db.bootstrap.CHUNK_SIZE", 2):
            chunked_upsert(client, "model_runs", records, on_conflict="run_id")

        calls = client.tables["model_runs"].calls
        self.assertEqual([len(call["records"]) for call in calls], [2, 2, 1])
        self.assertTrue(all(call["kwargs"]["returning"] == ReturnMethod.minimal for call in calls))


if __name__ == "__main__":
    unittest.main()
