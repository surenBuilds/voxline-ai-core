"""
Architecture-level tests for Voxline AI Core.

Verifies that canonical imports work, legacy compatibility is preserved
where retained, and the overall package structure is correct.
"""

import unittest
import sys
import os
import tempfile
import torch


class TestCanonicalImports(unittest.TestCase):
    """Verify all canonical package imports succeed."""

    def test_import_model(self):
        from src.model import VoxlineTransformer, PositionalEncoding
        self.assertIsNotNone(VoxlineTransformer)
        self.assertIsNotNone(PositionalEncoding)

    def test_import_tokenizer(self):
        from src.tokenizer import BPETokenizer
        self.assertIsNotNone(BPETokenizer)

    def test_import_training(self):
        from src.training import Trainer, TrainingConfig, LanguageModelDataset
        self.assertIsNotNone(Trainer)
        self.assertIsNotNone(TrainingConfig)
        self.assertIsNotNone(LanguageModelDataset)

    def test_import_inference(self):
        from src.inference import TextGenerator, GenerationConfig
        self.assertIsNotNone(TextGenerator)
        self.assertIsNotNone(GenerationConfig)

    def test_import_providers(self):
        from src.providers import (
            AIProvider, GenerationConfig, ProviderHealth,
            ProviderStatus, LocalVoxlineProvider, LocalTransformersProvider,
            ProviderFactory,
        )
        self.assertIsNotNone(AIProvider)
        self.assertIsNotNone(LocalVoxlineProvider)
        self.assertIsNotNone(LocalTransformersProvider)
        self.assertIsNotNone(ProviderFactory)

    def test_import_memory(self):
        from src.memory import MemoryStore, ConversationMemory, MemoryEntry
        self.assertIsNotNone(MemoryStore)
        self.assertIsNotNone(ConversationMemory)
        self.assertIsNotNone(MemoryEntry)

    def test_import_tools(self):
        from src.tools import ToolRegistry, Tool, Calculator, FileReadTool, FileWriteTool
        self.assertIsNotNone(ToolRegistry)
        self.assertIsNotNone(Calculator)

    def test_import_planner(self):
        from src.planner import Planner, ReasoningEngine, Plan, Step
        self.assertIsNotNone(Planner)
        self.assertIsNotNone(ReasoningEngine)

    def test_import_agent(self):
        from src.agent import AutonomousAgent, AgentState
        self.assertIsNotNone(AutonomousAgent)
        self.assertIsNotNone(AgentState)

    def test_import_business(self):
        from src.business import BusinessAgent, BusinessPlan, BusinessPlanStep
        self.assertIsNotNone(BusinessAgent)
        self.assertIsNotNone(BusinessPlan)

    def test_import_config(self):
        from src.config import ModelConfig, ModelType, VoxlineConfig
        self.assertIsNotNone(ModelConfig)
        self.assertIsNotNone(ModelType)
        self.assertIsNotNone(VoxlineConfig)

    def test_import_attention(self):
        from src.attention import ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention
        self.assertIsNotNone(ScaledDotProductAttention)
        self.assertIsNotNone(MultiHeadAttention)
        self.assertIsNotNone(CausalSelfAttention)

    def test_import_api(self):
        from src.api import ConversationalAI
        self.assertIsNotNone(ConversationalAI)

    def test_import_errors(self):
        from src.errors import (
            VoxlineError, ModelError, ModelLoadError, ModelInferenceError,
            CheckpointIncompatibilityError, ProviderNotFoundError,
            ToolNotFoundError, ToolExecutionError,
            ConfigError, TrainingError, AgentError,
        )
        self.assertTrue(issubclass(ModelError, VoxlineError))
        self.assertTrue(issubclass(CheckpointIncompatibilityError, VoxlineError))
        self.assertTrue(issubclass(ProviderNotFoundError, VoxlineError))

    def test_import_checkpoint(self):
        from src.checkpoint import CheckpointLoader, CheckpointIncompatibilityError
        self.assertIsNotNone(CheckpointLoader)

    def test_import_logging(self):
        from src.logging import setup_logging, get_logger, StructuredLogger
        self.assertIsNotNone(setup_logging)
        self.assertIsNotNone(get_logger)


class TestLegacyCompatibility(unittest.TestCase):
    """Verify legacy modules are accessible from src/legacy/."""

    def test_legacy_model(self):
        from src.legacy.model import VoxlineModel
        self.assertIsNotNone(VoxlineModel)

    def test_legacy_tokenizer(self):
        from src.legacy.tokenizer import SimpleTokenizer
        self.assertIsNotNone(SimpleTokenizer)

    def test_legacy_config(self):
        from src.legacy.config import TrainingConfig
        self.assertIsNotNone(TrainingConfig)


class TestModelArchitecture(unittest.TestCase):
    """Verify model can be instantiated and run."""

    def test_model_instantiation(self):
        from src.model.transformer import VoxlineTransformer
        model = VoxlineTransformer(
            vocab_size=200, d_model=64, num_layers=2,
            num_heads=4, d_ff=128, max_seq_len=32,
        )
        self.assertGreater(model.get_num_parameters(), 0)

    def test_model_forward(self):
        from src.model.transformer import VoxlineTransformer
        model = VoxlineTransformer(
            vocab_size=200, d_model=64, num_layers=2,
            num_heads=4, d_ff=128, max_seq_len=32,
        )
        inp = torch.randint(0, 200, (1, 10))
        out = model(inp)
        self.assertEqual(out.shape, (1, 10, 200))

    def test_model_generate(self):
        from src.model.transformer import VoxlineTransformer
        model = VoxlineTransformer(
            vocab_size=200, d_model=64, num_layers=2,
            num_heads=4, d_ff=128, max_seq_len=32,
        )
        inp = torch.randint(0, 200, (1, 5))
        out = model.generate(inp, max_new_tokens=5, temperature=1.0, pad_token_id=0)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 10)


class TestTokenizerArchitecture(unittest.TestCase):
    """Verify tokenizer can be instantiated and run."""

    def test_tokenizer_fit_encode_decode(self):
        from src.tokenizer.bpe import BPETokenizer
        tok = BPETokenizer(vocab_size=200)
        tok.fit(["hello world test", "hello world example"])
        ids = tok.encode("hello world")
        text = tok.decode(ids)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(text, str)
        self.assertTrue(len(ids) > 0)

    def test_tokenizer_save_load(self):
        from src.tokenizer.bpe import BPETokenizer
        tok = BPETokenizer(vocab_size=200)
        tok.fit(["hello world test"])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = BPETokenizer()
            tok2.load(path)
            ids1 = tok.encode("hello")
            ids2 = tok2.encode("hello")
            self.assertEqual(ids1, ids2)
        finally:
            os.unlink(path)


class TestMemoryArchitecture(unittest.TestCase):
    """Verify memory system works end-to-end."""

    def test_memory_store_crud(self):
        from src.memory.memory import MemoryStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            ms = MemoryStore(db_path=path)
            mid = ms.add_memory("test content", memory_type="episodic")
            entry = ms.get_memory(mid)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.content, "test content")
            ms.update_memory(mid, "updated content")
            entry = ms.get_memory(mid)
            self.assertEqual(entry.content, "updated content")
            ms.delete_memory(mid)
            entry = ms.get_memory(mid)
            self.assertIsNone(entry)
            ms.close()
        finally:
            os.unlink(path)


class TestToolsArchitecture(unittest.TestCase):
    """Verify tool system works."""

    def test_tool_registry_has_default_tools(self):
        from src.tools.tools import ToolRegistry
        tr = ToolRegistry(workspace_root=".")
        tools = tr.list_tools()
        self.assertIn("calculator", tools)
        self.assertIn("read_file", tools)
        self.assertIn("write_file", tools)
        self.assertIn("list_directory", tools)

    def test_calculator_executes(self):
        from src.tools.tools import ToolRegistry
        tr = ToolRegistry(workspace_root=".")
        result = tr.execute_tool("calculator", expression="2+2")
        self.assertEqual(result, 4.0)


class TestProviderArchitecture(unittest.TestCase):
    """Verify provider abstraction works."""

    def test_provider_factory_class_exists(self):
        from src.providers.factory import ProviderFactory
        self.assertIsNotNone(ProviderFactory)

    def test_provider_base_abc(self):
        from src.providers.base import AIProvider, GenerationConfig, ProviderHealth, ProviderStatus
        self.assertTrue(hasattr(AIProvider, 'generate'))
        self.assertTrue(hasattr(AIProvider, 'stream'))
        self.assertTrue(hasattr(AIProvider, 'health_check'))


class TestConfigArchitecture(unittest.TestCase):
    """Verify configuration system works."""

    def test_model_config_creation(self):
        from src.config.model_config import ModelConfig, ModelType
        cfg = ModelConfig.for_voxline_transformer(vocab_size=200, d_model=64)
        self.assertEqual(cfg.vocab_size, 200)
        self.assertEqual(cfg.d_model, 64)

    def test_model_config_from_dict(self):
        from src.config.model_config import ModelConfig
        data = {"vocab_size": 200, "d_model": 64, "max_seq_len": 32,
                "num_layers": 2, "num_heads": 4, "d_ff": 128}
        cfg = ModelConfig.from_dict(data)
        self.assertEqual(cfg.vocab_size, 200)

    def test_model_config_from_dict_ignores_unknown(self):
        from src.config.model_config import ModelConfig
        data = {"vocab_size": 200, "d_model": 64, "max_seq_len": 32,
                "unknown_field": "should_be_ignored"}
        cfg = ModelConfig.from_dict(data)
        self.assertEqual(cfg.vocab_size, 200)

    def test_voxline_config(self):
        from src.config.settings import VoxlineConfig
        cfg = VoxlineConfig()
        self.assertIsNotNone(cfg.ai_provider)


class TestPlannerArchitecture(unittest.TestCase):
    """Verify planner system works."""

    def test_planner_creates_plan(self):
        from src.planner.reasoning import Planner, PlanStatus
        p = Planner()
        plan = p.create_plan("test goal", ["step1", "step2"])
        self.assertEqual(len(plan.steps), 2)
        self.assertEqual(plan.status, PlanStatus.PENDING)

    def test_reasoning_engine(self):
        from src.planner.reasoning import ReasoningEngine
        re = ReasoningEngine()
        analysis = re.analyze_goal("test goal")
        self.assertIn("goal", analysis)


if __name__ == "__main__":
    unittest.main()
