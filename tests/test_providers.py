"""
Provider system tests.

Tests the AIProvider abstraction, factory, and provider implementations.
"""

import unittest
import asyncio
import torch
import os

from src.providers.base import (
    AIProvider, GenerationConfig, ProviderHealth, ProviderStatus, ModelInfo,
)
from src.providers.local_voxline import LocalVoxlineProvider
from src.providers.factory import ProviderFactory, _ensure_builtin_providers
from src.model.transformer import VoxlineTransformer
from src.tokenizer.bpe import BPETokenizer
from src.config.model_config import ModelConfig


def _make_small_model_and_tokenizer():
    tok = BPETokenizer(vocab_size=200)
    tok.fit(["hello world test", "test sentence here"])
    model = VoxlineTransformer(
        vocab_size=200, d_model=64, num_layers=2,
        num_heads=4, d_ff=128, max_seq_len=32,
    )
    return model, tok


class TestProviderBase(unittest.TestCase):

    def test_generation_config_defaults(self):
        cfg = GenerationConfig()
        self.assertEqual(cfg.max_tokens, 100)
        self.assertEqual(cfg.temperature, 1.0)
        self.assertTrue(cfg.do_sample)

    def test_generation_config_custom(self):
        cfg = GenerationConfig(max_tokens=50, temperature=0.5, top_k=10, top_p=0.9)
        self.assertEqual(cfg.max_tokens, 50)
        self.assertEqual(cfg.temperature, 0.5)

    def test_provider_status_enum(self):
        self.assertEqual(ProviderStatus.HEALTHY.value, "healthy")
        self.assertEqual(ProviderStatus.DEGRADED.value, "degraded")
        self.assertEqual(ProviderStatus.UNAVAILABLE.value, "unavailable")

    def test_model_info_dataclass(self):
        info = ModelInfo(model_id="m", provider_id="p", model_type="native", parameters=1000)
        self.assertEqual(info.model_id, "m")
        self.assertEqual(info.parameters, 1000)
        self.assertEqual(info.extra, {})

    def test_provider_health_dataclass(self):
        h = ProviderHealth(status=ProviderStatus.HEALTHY, message="ok", response_time_ms=10.5)
        self.assertEqual(h.status, ProviderStatus.HEALTHY)


class TestLocalVoxlineProvider(unittest.TestCase):

    def setUp(self):
        self.model, self.tokenizer = _make_small_model_and_tokenizer()
        self.config = ModelConfig(
            model_type="voxline_transformer", model_version="0.4.0",
            vocab_size=200, d_model=64, max_seq_len=32,
            num_layers=2, num_heads=4, d_ff=128,
        )
        self.provider = LocalVoxlineProvider(
            self.model, self.tokenizer, self.config, device="cpu"
        )

    def test_provider_id(self):
        self.assertEqual(self.provider.provider_id, "local_voxline")

    def test_model_id(self):
        self.assertIn("voxline", self.provider.model_id)

    def test_supports_streaming(self):
        self.assertTrue(self.provider.supports_streaming)

    def test_health_check(self):
        h = asyncio.run(self.provider.health_check())
        self.assertIn(h.status, (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED))

    def test_generate(self):
        cfg = GenerationConfig(max_tokens=10, temperature=1.0)
        result = asyncio.run(self.provider.generate("hello", cfg))
        self.assertIsInstance(result, str)

    def test_chat(self):
        cfg = GenerationConfig(max_tokens=10, temperature=1.0)
        messages = [{"role": "user", "content": "hello"}]
        result = asyncio.run(self.provider.chat(messages, cfg))
        self.assertIsInstance(result, str)

    def test_get_model_info(self):
        info = self.provider.get_model_info()
        self.assertEqual(info.provider_id, "local_voxline")
        self.assertEqual(info.model_type, "native")
        self.assertIsNotNone(info.parameters)
        self.assertGreater(info.parameters, 0)
        self.assertEqual(info.vocab_size, 200)

    def test_stream(self):
        cfg = GenerationConfig(max_tokens=5, temperature=1.0)
        tokens = []
        async def collect():
            async for t in self.provider.stream("hello", cfg):
                tokens.append(t)
        asyncio.run(collect())
        self.assertTrue(len(tokens) > 0)


class TestQwenProvider(unittest.TestCase):

    MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"

    def setUp(self):
        if not os.path.exists(self.MODEL_PATH):
            self.skipTest(f"Qwen model not found at {self.MODEL_PATH}")

    def test_import(self):
        from src.providers.qwen_provider import QwenProvider
        self.assertIsNotNone(QwenProvider)

    def test_provider_id(self):
        from src.providers.qwen_provider import QwenProvider
        p = QwenProvider(self.MODEL_PATH, device="cpu")
        self.assertEqual(p.provider_id, "qwen")

    def test_model_id(self):
        from src.providers.qwen_provider import QwenProvider
        p = QwenProvider(self.MODEL_PATH, device="cpu")
        self.assertIn("Qwen", p.model_id)

    def test_health_check(self):
        from src.providers.qwen_provider import QwenProvider
        p = QwenProvider(self.MODEL_PATH, device="cpu")
        h = asyncio.run(p.health_check())
        self.assertEqual(h.status, ProviderStatus.HEALTHY)

    def test_generate(self):
        from src.providers.qwen_provider import QwenProvider
        p = QwenProvider(self.MODEL_PATH, device="cpu")
        cfg = GenerationConfig(max_tokens=20, temperature=0.7)
        result = asyncio.run(p.generate("Hello", cfg))
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_chat(self):
        from src.providers.qwen_provider import QwenProvider
        p = QwenProvider(self.MODEL_PATH, device="cpu")
        cfg = GenerationConfig(max_tokens=20, temperature=0.7)
        messages = [{"role": "user", "content": "Say hello"}]
        result = asyncio.run(p.chat(messages, cfg))
        self.assertIsInstance(result, str)

    def test_get_model_info(self):
        from src.providers.qwen_provider import QwenProvider
        p = QwenProvider(self.MODEL_PATH, device="cpu")
        info = p.get_model_info()
        self.assertEqual(info.provider_id, "qwen")
        self.assertEqual(info.model_type, "huggingface")
        self.assertIsNotNone(info.parameters)
        self.assertGreater(info.parameters, 0)


class TestProviderFactory(unittest.TestCase):

    def test_ensure_builtin_providers(self):
        _ensure_builtin_providers()
        available = ProviderFactory.get_available_providers()
        self.assertIn("native", available)
        self.assertIn("qwen", available)

    def test_register_custom_provider(self):
        class DummyProvider(AIProvider):
            @property
            def provider_id(self): return "dummy"
            @property
            def model_id(self): return "dummy_model"
            @property
            def supports_streaming(self): return False
            async def generate(self, prompt, config): return "dummy"
            async def health_check(self): return ProviderHealth(ProviderStatus.HEALTHY, "ok")

        ProviderFactory.register_provider("dummy", DummyProvider)
        self.assertIn("dummy", ProviderFactory.get_available_providers())

    def test_create_native_provider(self):
        model, tokenizer = _make_small_model_and_tokenizer()
        config = ModelConfig(
            model_type="voxline_transformer", model_version="0.4.0",
            vocab_size=200, d_model=64, max_seq_len=32,
            num_layers=2, num_heads=4, d_ff=128,
        )
        from src.config.settings import VoxlineConfig
        vc = VoxlineConfig.__new__(VoxlineConfig)
        vc._config = {"AI_PROVIDER": "native", "AI_DEVICE": "cpu"}
        provider = ProviderFactory.create(
            vc, tokenizer=tokenizer, model=model, model_config=config
        )
        self.assertIsInstance(provider, LocalVoxlineProvider)

    def test_create_unknown_provider_raises(self):
        from src.config.settings import VoxlineConfig
        from src.errors import ProviderNotFoundError
        vc = VoxlineConfig.__new__(VoxlineConfig)
        vc._config = {"AI_PROVIDER": "nonexistent", "AI_DEVICE": "cpu"}
        with self.assertRaises(ProviderNotFoundError):
            ProviderFactory.create(vc)


class TestProviderInterface(unittest.TestCase):
    """Verify providers satisfy the AIProvider contract."""

    def test_local_voxline_is_aiprovider(self):
        self.assertTrue(issubclass(LocalVoxlineProvider, AIProvider))

    def test_qwen_is_aiprovider(self):
        from src.providers.qwen_provider import QwenProvider
        self.assertTrue(issubclass(QwenProvider, AIProvider))

    def test_chat_default_implementation(self):
        model, tokenizer = _make_small_model_and_tokenizer()
        config = ModelConfig(
            model_type="voxline_transformer", model_version="0.4.0",
            vocab_size=200, d_model=64, max_seq_len=32,
            num_layers=2, num_heads=4, d_ff=128,
        )
        provider = LocalVoxlineProvider(model, tokenizer, config, device="cpu")
        cfg = GenerationConfig(max_tokens=10)
        messages = [{"role": "user", "content": "hi"}]
        result = asyncio.run(provider.chat(messages, cfg))
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
