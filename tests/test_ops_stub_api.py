import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from connectors.ops_stub_api.main import app


class OpsStubApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"OPS_DATA_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.client = TestClient(app)

    def test_pause_then_unpause_updates_state_and_audit(self):
        r = self.client.post("/pause_item", json={"type": "pause_item", "item": "Chicken Burger", "operator": "demo"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["state"]["items"], {"Chicken Burger": "paused"})
        self.client.post("/unpause_item", json={"type": "unpause_item", "item": "Chicken Burger", "operator": "demo"})
        self.assertEqual(self.client.get("/state").json()["items"], {"Chicken Burger": "active"})
        audit = self.client.get("/audit").json()
        self.assertEqual(audit["count"], 2)
        self.assertTrue(audit["entries"][0]["ts"].endswith("Z"))

    def test_update_hours_records_the_details(self):
        r = self.client.post("/update_hours", json={"type": "update_hours", "details": "Sun 10-16", "operator": "demo"})
        self.assertEqual(r.json()["state"]["hours_note"], "Sun 10-16")

    def test_missing_item_and_mismatched_type_are_rejected(self):
        self.assertEqual(self.client.post("/pause_item", json={"type": "pause_item", "operator": "d"}).status_code, 400)
        r = self.client.post("/pause_item", json={"type": "update_hours", "item": "X", "operator": "d"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.get("/audit").json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
