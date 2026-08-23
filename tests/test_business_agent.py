import tempfile
import unittest
from pathlib import Path

from src.business.agent import BusinessAgent
from src.memory.memory import MemoryStore


class TestBusinessAgent(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(str(Path(self.temp_dir.name) / "business.db"))
        self.agent = BusinessAgent(self.memory)

    def tearDown(self):
        self.memory.close()
        self.temp_dir.cleanup()

    def test_stores_and_finds_business_knowledge(self):
        self.agent.remember("Our target customers are Armenian online retailers.", ["audience"])
        results = self.agent.search_knowledge("Armenian")
        self.assertEqual(len(results), 1)
        self.assertIn("retailers", results[0]["content"])

    def test_creates_a_reviewable_plan(self):
        plan = self.agent.create_plan("Increase sales for our service")
        self.assertEqual(plan.goal, "Increase sales for our service")
        self.assertEqual(len(plan.steps), 4)
        self.assertIn("վաճառ", plan.steps[1].title.lower())


if __name__ == "__main__":
    unittest.main()
