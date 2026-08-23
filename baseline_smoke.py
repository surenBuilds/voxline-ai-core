import sys
sys.stdout.reconfigure(encoding='utf-8')
import traceback
import os

results = []

def test(name, fn):
    try:
        fn()
        results.append((name, 'PASS', ''))
        print(f'  [PASS] {name}')
    except Exception as e:
        results.append((name, 'FAIL', str(e)))
        print(f'  [FAIL] {name}: {e}')
        traceback.print_exc()

# 1. TOKENIZER
print('=== 1. TOKENIZER ===')
def test_tokenizer():
    from src.tokenizer.bpe import BPETokenizer
    tok = BPETokenizer(vocab_size=200)
    tok.fit(['hello world test', 'hello world example'])
    ids = tok.encode('hello world')
    text = tok.decode(ids)
    assert len(ids) > 0
    assert isinstance(text, str)
    tok.save('_tmp_tok.json')
    tok2 = BPETokenizer()
    tok2.load('_tmp_tok.json')
    ids2 = tok2.encode('hello world')
    assert ids == ids2
    os.remove('_tmp_tok.json')
    print(f'    vocab_size={tok.get_vocab_size()}, encode={ids}, decode={repr(text)}')
test('tokenizer_encode_decode', test_tokenizer)

# 2. MODEL FORWARD
print('=== 2. MODEL ===')
def test_model():
    import torch
    from src.model.transformer import VoxlineTransformer
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    n = model.get_num_parameters()
    inp = torch.randint(0, 200, (1, 10))
    out = model(inp)
    assert out.shape == (1, 10, 200)
    print(f'    params={n}, output_shape={out.shape}')
test('model_forward', test_model)

# 3. MODEL GENERATION
print('=== 3. GENERATION ===')
def test_generation():
    import torch
    from src.model.transformer import VoxlineTransformer
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    inp = torch.randint(0, 200, (1, 5))
    out = model.generate(inp, max_new_tokens=10, temperature=1.0, pad_token_id=0)
    assert out.shape[0] == 1
    assert out.shape[1] == 15
    print(f'    input_len=5, output_len={out.shape[1]}')
test('model_generation', test_generation)

# 4. CHECKPOINT
print('=== 4. CHECKPOINT ===')
def test_checkpoint():
    import torch
    from src.model.transformer import VoxlineTransformer
    from src.config.model_config import ModelConfig
    from src.checkpoint import CheckpointLoader
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    cfg = ModelConfig.for_voxline_transformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    CheckpointLoader.save_checkpoint(model.state_dict(), cfg, '_tmp_ckpt.pt')
    loaded_state, loaded_cfg = CheckpointLoader.load_checkpoint('_tmp_ckpt.pt', cfg)
    model2 = VoxlineTransformer(**loaded_cfg.to_model_kwargs())
    model2.load_state_dict(loaded_state)
    os.remove('_tmp_ckpt.pt')
    os.remove('_tmp_ckpt.config.json')
    print(f'    roundtrip OK, params={model2.get_num_parameters()}')
test('checkpoint_save_load', test_checkpoint)

# 5. INFERENCE GENERATOR
print('=== 5. INFERENCE ===')
def test_generator():
    import torch
    from src.model.transformer import VoxlineTransformer
    from src.tokenizer.bpe import BPETokenizer
    from src.inference.generator import TextGenerator, GenerationConfig
    tok = BPETokenizer(vocab_size=200)
    tok.fit(['hello world', 'test sentence'])
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    gen = TextGenerator(model, tok, device='cpu')
    cfg = GenerationConfig(max_new_tokens=10, temperature=1.0)
    result = gen.generate('hello', config=cfg, return_text=True)
    assert isinstance(result, str)
    print(f'    generate text: {repr(result[:80])}')
test('inference_generate', test_generator)

# 6. MEMORY
print('=== 6. MEMORY ===')
def test_memory():
    from src.memory.memory import MemoryStore, ConversationMemory
    db = '_tmp_mem.db'
    ms = MemoryStore(db_path=db)
    mid = ms.add_memory('test memory content', memory_type='episodic', source='test')
    search_results = ms.search_memories('test')
    assert len(search_results) >= 1
    cm = ConversationMemory(ms)
    cm.add_message('user', 'hello')
    cm.add_message('assistant', 'hi there')
    ctx = cm.get_context()
    assert 'hello' in ctx
    ms.close()
    os.remove(db)
    print(f'    search_results={len(search_results)}, context_len={len(ctx)}')
test('memory_store', test_memory)

# 7. TOOLS
print('=== 7. TOOLS ===')
def test_tools():
    from src.tools.tools import ToolRegistry
    tr = ToolRegistry(workspace_root='.')
    tools = tr.list_tools()
    assert 'calculator' in tools
    result = tr.execute_tool('calculator', expression='2+2')
    assert result == 4.0
    print(f'    tools={list(tools.keys())}, calc(2+2)={result}')
test('tool_registry', test_tools)

# 8. PLANNER
print('=== 8. PLANNER ===')
def test_planner():
    from src.planner.reasoning import Planner, ReasoningEngine, PlanStatus
    p = Planner()
    plan = p.create_plan('test goal', ['step1', 'step2'])
    assert len(plan.steps) == 2
    assert plan.status == PlanStatus.PENDING
    re = ReasoningEngine()
    analysis = re.analyze_goal('test goal')
    assert isinstance(analysis, dict)
    print(f'    steps={len(plan.steps)}, status={plan.status}')
test('planner', test_planner)

# 9. AGENT
print('=== 9. AGENT ===')
def test_agent():
    import torch
    from src.model.transformer import VoxlineTransformer
    from src.tokenizer.bpe import BPETokenizer
    from src.agent.agent import AutonomousAgent, AgentState
    tok = BPETokenizer(vocab_size=200)
    tok.fit(['test'])
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    agent = AutonomousAgent(model, tok, device='cpu')
    assert agent.get_state() == AgentState.IDLE
    agent.set_goal('test goal')
    tools = agent.get_tools()
    print(f'    state={agent.get_state()}, tools={tools}')
test('agent', test_agent)

# 10. BUSINESS AGENT
print('=== 10. BUSINESS AGENT ===')
def test_business():
    from src.memory.memory import MemoryStore
    from src.business.agent import BusinessAgent
    db = '_tmp_biz.db'
    ms = MemoryStore(db_path=db)
    ba = BusinessAgent(ms)
    plan = ba.create_plan('test business goal')
    assert len(plan.steps) > 0
    ba.remember('test business knowledge', tags=['test'])
    bresults = ba.search_knowledge('test')
    assert len(bresults) >= 1
    ms.close()
    os.remove(db)
    print(f'    plan_steps={len(plan.steps)}, knowledge_results={len(bresults)}')
test('business_agent', test_business)

# 11. PROVIDER
print('=== 11. PROVIDER ===')
def test_provider():
    import torch
    from src.model.transformer import VoxlineTransformer
    from src.tokenizer.bpe import BPETokenizer
    from src.config.model_config import ModelConfig
    from src.providers.local_voxline import LocalVoxlineProvider
    import asyncio
    tok = BPETokenizer(vocab_size=200)
    tok.fit(['test'])
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    cfg = ModelConfig.for_voxline_transformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    from src.providers.base import GenerationConfig as ProviderGenConfig
    prov = LocalVoxlineProvider(model, tok, cfg, device='cpu')
    health = asyncio.run(prov.health_check())
    assert health.status.value in ('healthy', 'degraded')
    pcfg = ProviderGenConfig(max_tokens=10, temperature=1.0)
    result = asyncio.run(prov.generate('test', config=pcfg))
    assert isinstance(result, str)
    print(f'    health={health.status.value}, gen={repr(result[:60])}')
test('provider', test_provider)

# 12. SETTINGS
print('=== 12. SETTINGS ===')
def test_settings():
    from src.config.settings import VoxlineConfig
    cfg = VoxlineConfig()
    assert cfg.ai_provider is not None
    print(f'    ai_provider={cfg.ai_provider}, device={cfg.ai_device}')
test('settings', test_settings)

# 13. CONVERSATIONAL AI
print('=== 13. CONVERSATIONAL AI ===')
def test_chat():
    import torch
    from src.model.transformer import VoxlineTransformer
    from src.tokenizer.bpe import BPETokenizer
    from src.api.chat import ConversationalAI
    tok = BPETokenizer(vocab_size=200)
    tok.fit(['hello world', 'test sentence'])
    model = VoxlineTransformer(vocab_size=200, d_model=64, num_layers=2, num_heads=4, d_ff=128, max_seq_len=32)
    chat = ConversationalAI(model, tok, device='cpu')
    resp = chat.chat('hello', include_memory=False, max_new_tokens=10)
    assert isinstance(resp, str)
    chat.clear_conversation()
    print(f'    chat response: {repr(resp[:80])}')
test('conversational_ai', test_chat)

# 14. BEST MODEL CHECKPOINT
print('=== 14. BEST MODEL CHECKPOINT ===')
def test_best_model():
    import torch
    from src.model.transformer import VoxlineTransformer
    from src.config.model_config import ModelConfig
    ckpt_path = 'checkpoints/v0_4/best_model.pt'
    if not os.path.exists(ckpt_path):
        print('    SKIP: no best_model.pt found')
        return
    data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in data
    assert 'config' in data
    cfg_raw = data['config']
    if isinstance(cfg_raw, dict):
        # Filter to only ModelConfig-accepted fields
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(ModelConfig)}
        filtered = {k: v for k, v in cfg_raw.items() if k in valid_keys}
        cfg = ModelConfig.from_dict(filtered)
    else:
        cfg = cfg_raw
    model = VoxlineTransformer(**cfg.to_model_kwargs())
    model.load_state_dict(data['model_state_dict'])
    n = model.get_num_parameters()
    history = data.get('training_history', {})
    print(f'    loaded best_model.pt: params={n}, step={data.get("global_step", "?")}, history_keys={list(history.keys()) if isinstance(history, dict) else "?"}')
test('best_model_checkpoint', test_best_model)

print()
print('=' * 50)
passed = sum(1 for _, s, _ in results if s == 'PASS')
failed = sum(1 for _, s, _ in results if s == 'FAIL')
print(f'RESULTS: {passed} passed, {failed} failed, {len(results)} total')
print('=' * 50)
for name, status, err in results:
    line = f'  {status}: {name}'
    if err:
        line += f' -- {err[:60]}'
    print(line)
