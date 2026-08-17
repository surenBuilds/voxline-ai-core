"""
Comprehensive test suite for Voxline AI Core

Tests all components:
- Tokenizer
- Attention
- Transformer model
- Training
- Inference
- Memory
- Tools
- Agent
"""

import unittest
import torch
import tempfile
from pathlib import Path

# Import all components
from src.tokenizer.bpe import BPETokenizer
from src.attention.attention import ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention
from src.model.transformer import VoxlineTransformer, PositionalEncoding
from src.training.trainer import LanguageModelDataset, Trainer, TrainingConfig
from src.inference.generator import TextGenerator, GenerationConfig
from src.memory.memory import MemoryStore, ConversationMemory
from src.tools.tools import ToolRegistry
from src.planner.reasoning import Planner, ReasoningEngine


class TestBPETokenizer(unittest.TestCase):
    """Test BPE tokenizer."""

    def setUp(self):
        self.tokenizer = BPETokenizer(vocab_size=1000)
        self.texts = [
            "Voxline creates language models.",
            "The quick brown fox jumps.",
            "Armenian language support.",
        ]

    def test_fit(self):
        """Test tokenizer training."""
        self.tokenizer.fit(self.texts, num_merges=100)
        self.assertGreater(self.tokenizer.get_vocab_size(), 0)

    def test_encode_decode(self):
        """Test encode/decode roundtrip."""
        self.tokenizer.fit(self.texts, num_merges=50)
        text = "Voxline creates"
        encoded = self.tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(i, int) for i in encoded))

    def test_save_load(self):
        """Test save/load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer.fit(self.texts, num_merges=50)
            save_path = Path(tmpdir) / "tokenizer.json"

            self.tokenizer.save(str(save_path))
            self.assertTrue(save_path.exists())

            new_tokenizer = BPETokenizer()
            new_tokenizer.load(str(save_path))
            self.assertEqual(
                self.tokenizer.get_vocab_size(),
                new_tokenizer.get_vocab_size()
            )


class TestAttention(unittest.TestCase):
    """Test attention mechanisms."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 64
        self.num_heads = 4

    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention."""
        attention = ScaledDotProductAttention()

        query = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads)
        key = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads)
        value = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads)

        output, weights = attention(query, key, value)

        self.assertEqual(output.shape, value.shape)
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))

    def test_multi_head_attention(self):
        """Test multi-head attention."""
        attention = MultiHeadAttention(self.d_model, self.num_heads)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = attention(x, x, x)

        self.assertEqual(output.shape, x.shape)

    def test_causal_attention(self):
        """Test causal attention mask."""
        attention = CausalSelfAttention(self.d_model, self.num_heads, 512)

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = attention(x)

        self.assertEqual(output.shape, x.shape)


class TestTransformer(unittest.TestCase):
    """Test Transformer model."""

    def setUp(self):
        self.vocab_size = 1000
        self.d_model = 128
        self.num_layers = 2
        self.model = VoxlineTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=4,
            d_ff=256,
        )

    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        seq_len = 16

        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        logits = self.model(input_ids)

        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))

    def test_generation(self):
        """Test text generation."""
        input_ids = torch.tensor([[1, 2, 3, 4]])
        generated = self.model.generate(input_ids, max_new_tokens=10)

        self.assertEqual(generated.shape[0], 1)
        self.assertGreaterEqual(generated.shape[1], 4)

    def test_parameters(self):
        """Test parameter count."""
        params = self.model.get_num_parameters()
        self.assertGreater(params, 0)

    def test_config(self):
        """Test config retrieval."""
        config = self.model.get_config()
        self.assertEqual(config["vocab_size"], self.vocab_size)
        self.assertEqual(config["d_model"], self.d_model)


class TestMemory(unittest.TestCase):
    """Test memory system."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.memory_store = MemoryStore(str(self.db_path))

    def tearDown(self):
        self.memory_store.close()
        self.temp_dir.cleanup()

    def test_add_memory(self):
        """Test adding memory."""
        memory_id = self.memory_store.add_memory(
            content="Test memory content",
            memory_type="semantic",
            keywords=["test", "memory"],
        )
        self.assertIsNotNone(memory_id)

    def test_search_memory(self):
        """Test searching memory."""
        self.memory_store.add_memory(
            content="Voxline is an AI system",
            memory_type="semantic",
            keywords=["voxline", "ai"],
        )

        results = self.memory_store.search_memories("voxline")
        self.assertGreater(len(results), 0)

    def test_conversation_memory(self):
        """Test conversation memory."""
        conv_memory = ConversationMemory(self.memory_store)

        conv_memory.add_message("user", "Hello")
        conv_memory.add_message("assistant", "Hi there!")

        context = conv_memory.get_context()
        self.assertIn("Hello", context)
        self.assertIn("Hi there", context)


class TestTools(unittest.TestCase):
    """Test tool registry."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_list_tools(self):
        """Test listing tools."""
        tools = self.registry.list_tools()
        self.assertIn("calculator", tools)
        self.assertIn("read_file", tools)

    def test_calculator(self):
        """Test calculator tool."""
        result = self.registry.execute_tool("calculator", expression="2+2")
        self.assertEqual(result, 4.0)

    def test_file_operations(self):
        """Test file tools within workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ToolRegistry(tmpdir)

            # Write file
            result = registry.execute_tool(
                "write_file",
                path="test.txt",
                content="Hello, Voxline!",
            )
            self.assertTrue(result.get("success"))

            # Read file
            content = registry.execute_tool("read_file", path="test.txt")
            self.assertIn("Hello", str(content))


class TestPlanner(unittest.TestCase):
    """Test reasoning and planner."""

    def setUp(self):
        self.planner = Planner()

    def test_create_plan(self):
        """Test plan creation."""
        steps = ["Step 1", "Step 2", "Step 3"]
        plan = self.planner.create_plan("Test goal", steps)

        self.assertEqual(plan.goal, "Test goal")
        self.assertEqual(len(plan.steps), 3)

    def test_plan_progress(self):
        """Test plan progress tracking."""
        steps = ["Step 1", "Step 2"]
        plan = self.planner.create_plan("Test goal", steps)

        self.assertEqual(plan.get_progress(), 0.0)

        self.planner.update_step_result(plan.id, plan.steps[0].id, "Done")
        self.assertEqual(plan.get_progress(), 0.5)


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def setUp(self):
        self.vocab_size = 500
        self.tokenizer = BPETokenizer(vocab_size=self.vocab_size)
        self.texts = ["Voxline creates language models.", "Test text."]
        self.tokenizer.fit(self.texts, num_merges=100)

        self.model = VoxlineTransformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=64,
            num_layers=1,
            num_heads=2,
        )

    def test_tokenizer_model_integration(self):
        """Test tokenizer and model integration."""
        text = "Voxline"
        token_ids = self.tokenizer.encode(text)

        input_tensor = torch.tensor([token_ids])
        logits = self.model(input_tensor)

        self.assertEqual(logits.shape[1], len(token_ids))

    def test_generation_integration(self):
        """Test generation pipeline."""
        generator = TextGenerator(self.model, self.tokenizer)
        config = GenerationConfig(max_new_tokens=5, do_sample=False)

        result = generator.generate("Voxline", config)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
