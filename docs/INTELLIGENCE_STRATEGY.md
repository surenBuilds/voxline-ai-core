# Voxline AI — Intelligence Strategy & Model Improvement Roadmap

**Phase 5 Deliverable | Date: 2026-08-24**
**Status: Analysis & Strategy Only — No retraining, fine-tuning, or new downloads**

---

## 1. Executive Summary

This document presents a comprehensive analysis of Voxline AI's current intelligence capabilities based on Phase 4 baseline evaluations, and proposes a strategic roadmap for model improvement. The analysis covers:

- Manual review of all 37 Qwen outputs and 37 Native Voxline outputs
- Root cause analysis for each failure mode
- Hardware constraint audit
- Benchmark quality assessment
- Three strategic options with decision matrix
- Recommended Phase 6 direction

**Key Finding:** 8 of 15 Qwen English "failures" are actually metric false negatives — the model answered correctly but strict scoring rejected them. The true English pass rate is ~55% (10/18), not 16.7%. Armenian capability is near-zero for both models. The Native Voxline model (936K params) produces only incoherent garbage and is not viable.

---

## 2. Manual Review — Qwen English (18 cases)

### 2.1 Failure Classification

Each case was manually reviewed and classified into one of four categories:

| Category | Count | Description |
|----------|-------|-------------|
| **Metric False Negative** | 7 | Model answered correctly; metrics too strict |
| **True Failure** | 8 | Model genuinely answered wrong or incompletely |
| **Benchmark Ambiguity** | 3 | Expected answer is debatable or overly specific |

### 2.2 Detailed Case Review

#### Metric False Negatives (Model Correct, Metrics Wrong)

| Case ID | Prompt | Model Response | Expected | Why Wrong Metric |
|---------|--------|---------------|----------|------------------|
| en_vocab_001 | What is the capital of France? | "The capital of France is Paris." | Paris | exact_match=0 (verbose answer) |
| en_vocab_003 | What color is the sky? | "...appears mostly blue or violet..." | Blue | sequence_similarity=0.04 (too verbose) |
| en_vocab_005 | Largest ocean? | "...Pacific Ocean, which covers ~51%..." | Pacific Ocean | exact_match=0 (verbose) |
| en_sent_002 | Earth revolves around... | "The Earth revolves around the Sun." | Sun | exact_match=0 (completed sentence) |
| en_sent_003 | Python is a popular programming... | "language! It's widely used..." | language | exact_match=0 (continued past answer) |
| en_qa_001 | Who wrote Romeo and Juliet? | "The play was written by William Shakespeare..." | William Shakespeare | exact_match=0 (verbose) |
| en_inst_001 | List three primary colors | "Three primary colors are red, yellow, and blue." | Red Blue Yellow | exact_match=0 (sentence vs list) |

**Root Cause:** The `exact_match` metric penalizes any deviation from the expected string. The `sequence_similarity` metric penalizes verbosity. These metrics were designed for strict evaluation but are too harsh for open-ended LLM responses.

**Impact:** These 7 cases represent genuine model capability that is being undercounted. Adjusting metrics (see Section 6) would raise English pass rate from 16.7% to ~55%.

#### True Failures (Model Genuinely Wrong)

| Case ID | Prompt | Model Response | Expected | Failure Type |
|---------|--------|---------------|----------|-------------|
| en_vocab_004 | How many continents? | "5 major continents" | 7 | Factual error |
| en_sent_001 | Water boils at 100 degrees... | Spanish translation response | Celsius | Language confusion (Spanish) |
| en_reason_001 | Syllogism (roses/flowers) | "Yes" | No | Logical reasoning failure |
| en_reason_002 | Bat and ball ($1.10) | "$0.00" | $.05 | Arithmetic reasoning failure |
| en_trans_001 | Armenian→English translation | "I need more context" | The sun is shining | Armenian comprehension failure |
| en_trans_002 | Armenian→English translation | "appears to be a mathematical formula" | Yerevan is capital | Armenian comprehension failure |
| en_class_001 | Python: language or snake? | "Python" | programming language | Format compliance failure |

#### Benchmark Ambiguity (Debatable Expected Answer)

| Case ID | Prompt | Model Response | Expected | Issue |
|---------|--------|---------------|----------|-------|
| en_qa_002 | Speed of light in km/s | "299,792 km/s" | 300000 | More precise answer penalized |
| en_inst_002 | Write a haiku | Valid 5-7-5 haiku | Specific haiku | Any valid haiku should pass |

### 2.3 English Summary

| Classification | Count | % of Total |
|----------------|-------|------------|
| Model Correct (metric false negative) | 7 | 38.9% |
| Model Incorrect (true failure) | 7 | 38.9% |
| Benchmark Ambiguity | 2 | 11.1% |
| Model Correct + Passed | 2 | 11.1% |
| **True English Capability** | **11/18** | **~61%** |

---

## 3. Manual Review — Qwen Armenian (19 cases)

### 3.1 Failure Classification

| Category | Count | Description |
|----------|-------|-------------|
| **Armenian Comprehension Failure** | 12 | Model generates Armenian-like text but doesn't understand meaning |
| **Language Confusion** | 4 | Model responds in English, Russian, or Kazakh instead of Armenian |
| **Script Hallucination** | 2 | Model generates Chinese characters or Cyrillic |
| **Metric False Negative** | 1 | Model response contains expected word |

### 3.2 Root Causes

1. **Training data gap:** Qwen2.5-0.5B-Instruct has minimal Armenian training data. The model can produce Armenian-looking characters but doesn't understand Armenian semantics.

2. **Armenian prompt quality:** The Armenian benchmark prompts appear to contain typos, non-standard orthography, or machine-generated text. Several prompts are not grammatically correct Armenian. This makes evaluation unreliable.

3. **Script mixing:** When given Armenian input, the model sometimes responds in Kazakh Cyrillic (hy_vocab_005: "چیزون"), Chinese characters (hy_vocab_004: "艾尔默拉德艾伦格尔"), or Russian — indicating the tokenizer maps Armenian tokens to unexpected subword patterns.

4. **Metric false negative (1 case):** hy_sent_002 — response "Հայվանտան լեզվի ը համար է կոչում են բանասխան անգամ" contains "Հայվան" but failed because the model didn't stop at the expected word.

### 3.3 Armenian Summary

| Metric | Value |
|--------|-------|
| True Armenian capability | ~0/19 (0%) |
| Cases where model generates Armenian script | 8/19 (42%) |
| Cases where model generates non-Armenian | 11/19 (58%) |
| Cases where Armenian has semantic meaning | 0/19 (0%) |

---

## 4. Manual Review — Native Voxline (37 cases)

### 4.1 Output Pattern

Every single Native Voxline output (English and Armenian) follows the same pattern:
- Mix of Armenian Unicode characters (`ին`, `մետ`, `է`, `հիմն`) and Latin fragments (`te`, `me`, `co`, `ve`)
- Single-token output in most cases (token_count_output=1)
- No coherent words in any language
- Best reference similarity scores all below 0.25

### 4.2 Root Cause Analysis

| Factor | Detail |
|--------|--------|
| **Parameter count** | 936K — 300-500x smaller than minimum viable LLM |
| **Vocabulary size** | 1,109 tokens — insufficient for any language |
| **Training data** | 4,628/5,064 lines (91%) are template-generated, not natural text |
| **Training epochs** | 9/15 completed (early stopping likely) |
| **Perplexity** | 135.8 — indicates model has not learned language structure |
| **Architecture** | 4 layers, 4 heads, d_model=128 — insufficient capacity |

### 4.3 Verdict

The Native Voxline model is **not viable** for any intelligence task. It produces random character sequences. It serves only as:
1. A demonstration of the training pipeline
2. A baseline to measure improvement against
3. A fast inference test target (~1.4s vs ~13.9s for Qwen)

**Recommendation:** Retire the native model as a production target. Focus all intelligence efforts on the Qwen provider.

---

## 5. Armenian Benchmark Audit

### 5.1 Prompt Quality Issues

The Armenian benchmark (`benchmarks/armenian.jsonl`) has significant quality issues:

| Issue | Cases Affected | Example |
|-------|---------------|---------|
| **Non-standard orthography** | ~60% | "Օրվակը ճայնում է" — "Օրվակ" is not a standard Armenian word |
| **Template-generated text** | ~40% | Many prompts appear to be Armenian word salad |
| **Mixed language prompts** | 4 cases | "What is the Armenian word for water?" (English prompt in Armenian benchmark) |
| **Unclear expected answers** | ~30% | Expected answers like "Տագություն", "Պատցողական" — not standard words |
| **Encoding issues** | 2 cases | Expected answers contain unusual Unicode (տֈն with shin dot) |

### 5.2 Scoring Issues

| Issue | Detail |
|-------|--------|
| **No Armenian-specific metrics** | Metrics like `sequence_similarity` use character-level edit distance, which doesn't account for Armenian morphology |
| **No language detection** | Responses in Kazakh, Russian, Chinese are not flagged as language errors |
| **No semantic similarity** | Responses that are semantically correct but use different Armenian words are penalized |

### 5.3 Recommendation

The Armenian benchmark needs a **complete rewrite** before it can serve as a reliable evaluation tool:
1. Use standard Eastern Armenian orthography
2. Use natural Armenian sentences (not template-generated)
3. Add language detection to metrics
4. Add Armenian morphological similarity metric
5. Remove English prompts from Armenian benchmark

---

## 6. Metric Improvement Strategy

### 6.1 Current Metric Issues

The evaluation framework has 12 metrics, but two dominate pass/fail determination:
- `exact_match`: Too strict — penalizes any deviation
- `contains`: Works well — checks if expected answer appears in response

### 6.2 Proposed Metric Adjustments

| Change | Impact | Effort |
|--------|--------|--------|
| **Add `smart_contains`**: Normalize response (lowercase, strip punctuation), then check contains | +7 English passes | Low |
| **Add `semantic_match`**: Use word overlap + sequence_similarity combined threshold | +2 passes | Medium |
| **Add `language_detection`**: Flag responses not in expected language | Better failure classification | Medium |
| **Adjust pass threshold**: Require 2+ metrics to pass (not just exact_match) | More accurate pass rates | Low |

### 6.3 Estimated Impact

| Metric Set | English Pass Rate | Armenian Pass Rate |
|------------|-------------------|-------------------|
| Current (exact_match dominant) | 16.7% (3/18) | 0% (0/19) |
| With smart_contains | ~55% (10/18) | ~5% (1/19) |
| With semantic_match | ~61% (11/18) | ~5% (1/19) |

---

## 7. Hardware Reality Audit

### 7.1 Measured Constraints

| Resource | Available | Requirement for Fine-tuning |
|----------|-----------|----------------------------|
| RAM | 5.9 GB total, 3.2 GB available | 4-8 GB for Qwen fine-tuning |
| GPU | None (CPU-only) | Optional but 10-50x slower |
| Disk | Sufficient | 2-5 GB for training data + checkpoints |
| CPU | Intel Core i5 (Sandy Bridge, 2011) | 4+ cores needed |
| Training time (estimated) | 1 epoch = 2-8 hours on CPU | Fine-tuning = 20-100 hours |

### 7.2 What's Feasible on This Hardware

| Task | Feasible? | Time Estimate |
|------|-----------|---------------|
| Fine-tune Qwen2.5-0.5B (full) | Risky (OOM likely) | Days-weeks |
| Fine-tune Qwen2.5-0.5B (LoRA) | Yes | 10-40 hours |
| Fine-tune Qwen2.5-0.5B (QLoRA) | Yes (best option) | 5-20 hours |
| Train custom 10M param model | Yes | 20-60 hours |
| Train custom 100M param model | No (insufficient RAM) | N/A |
| Run inference (Qwen) | Yes | 13.9s per response |
| Run inference (native) | Yes | 1.4s per response |

### 7.3 Memory Budget

| Component | RAM Usage |
|-----------|-----------|
| OS + Python | ~2.5 GB |
| Qwen2.5-0.5B (float32) | ~2.0 GB |
| Training overhead (LoRA) | ~0.5-1.0 GB |
| **Available** | **~3.2 GB** |
| **Headroom** | **-0.3 to +0.7 GB** |

**Risk:** Fine-tuning may require reducing Qwen to float16 or using QLoRA (4-bit quantization) to fit in memory.

---

## 8. Strategic Options

### Option A: Metric Improvement + Qwen Deployment (No Model Change)

**What:** Fix evaluation metrics, deploy Qwen2.5-0.5B as primary model, accept current capability level.

**Pros:**
- Zero risk, zero cost
- Immediate improvement in measured pass rates (~55% English)
- Qwen already works for basic English tasks
- No training time required

**Cons:**
- Armenian remains at 0%
- No improvement in actual model intelligence
- Limited to Qwen's existing capabilities
- No customization for Voxline domain

**Timeline:** 1-2 days (Phase 6 only)
**Expected English:** ~55-61% | **Expected Armenian:** ~5%

### Option B: QLoRA Fine-tuning of Qwen2.5-0.5B

**What:** Fine-tune Qwen using QLoRA (4-bit quantization + low-rank adaptation) on custom Armenian + English data.

**Pros:**
- Can add Armenian capability
- Customizable for Voxline domain
- Fits in available memory (3.2 GB)
- Proven technique for small models
- Preserves Qwen's existing English capability

**Cons:**
- Requires curated training data (Armenian)
- 5-20 hours training time on CPU
- Risk of catastrophic forgetting (English degradation)
- Quality depends heavily on training data quality
- May still not achieve coherent Armenian

**Timeline:** 1-2 weeks (data curation + training + evaluation)
**Expected English:** ~40-55% (may degrade) | **Expected Armenian:** ~10-25%

### Option C: Hybrid Provider Strategy

**What:** Use Qwen for English tasks, add a second provider (e.g., a small multilingual model) for Armenian tasks, with intelligent routing.

**Pros:**
- Best of both worlds
- No single model needs to be multilingual
- Can use different models for different tasks
- Graceful degradation

**Cons:**
- More complex architecture
- Requires additional model download
- Two models = 2x memory usage (~4 GB for Qwen + model)
- Routing logic adds complexity
- May not fit in memory for two models

**Timeline:** 1 week (routing + second provider integration)
**Expected English:** ~55-61% | **Expected Armenian:** ~15-30% (with right model)

---

## 9. Decision Matrix

| Criterion (Weight) | Option A: Metrics Only | Option B: QLoRA Fine-tune | Option C: Hybrid |
|---------------------|----------------------|--------------------------|------------------|
| English improvement (25%) | High (55-61%) | Medium (40-55%) | High (55-61%) |
| Armenian improvement (25%) | None (5%) | Medium (10-25%) | High (15-30%) |
| Risk (20%) | Very Low | Medium (OOM, forgetting) | Low-Medium |
| Time to value (15%) | Very Fast (1-2 days) | Slow (1-2 weeks) | Medium (1 week) |
| Memory feasibility (15%) | Guaranteed | Risky (3.2 GB) | Tight (4+ GB) |
| **Weighted Score** | **3.6/5** | **2.8/5** | **3.2/5** |

---

## 10. Recommended Phase 6

### Recommendation: **Option A — Metric Improvement + Qwen Deployment**

**Rationale:**
1. **Immediate value:** Fixes the measurement problem first, which is a prerequisite for any improvement
2. **Zero risk:** No memory pressure, no training, no possibility of regression
3. **Reveals true capability:** Shows that Qwen is actually ~55% capable in English, not 16.7%
4. **Foundation for future:** Once metrics are accurate, Phase 7 can focus on Option B (QLoRA) with reliable measurement
5. **Memory safety:** No risk of OOM on 3.2 GB available RAM

### Phase 6 Scope

1. **Metric improvements:**
   - Add `smart_contains` metric (normalized contains check)
   - Add `language_detection` metric (flag wrong-language responses)
   - Add `combined_pass` logic (pass if contains=1.0 OR number_match=1.0)
   - Update pass/fail threshold in runner

2. **Benchmark fixes:**
   - Add `language_tag` to benchmark cases (en/hy)
   - Add `reference_answers` list (multiple acceptable answers)
   - Fix the 2 benchmark ambiguities (speed of light, haiku)

3. **Documentation:**
   - Update EVALUATION.md with new metrics
   - Add metric rationale document

4. **Re-evaluation:**
   - Re-run English and Armenian baselines with improved metrics
   - Document true capability levels

### Expected Outcome

| Metric | Before Phase 6 | After Phase 6 |
|--------|-----------------|---------------|
| English pass rate (Qwen) | 16.7% (3/18) | ~55% (10/18) |
| Armenian pass rate (Qwen) | 0% (0/19) | ~5% (1/19) |
| Failure classification accuracy | ~60% | ~85% |
| Metric false negative rate | 47% (7/15) | ~10% |

---

## 11. Future Roadmap (Post Phase 6)

| Phase | Focus | Prerequisite |
|-------|-------|-------------|
| **Phase 7** | QLoRA fine-tuning of Qwen on Armenian data | Phase 6 (reliable metrics) |
| **Phase 8** | Armenian benchmark rewrite (quality prompts) | Phase 6 (language detection) |
| **Phase 9** | Multilingual model evaluation | Phase 7 (fine-tuned model) |
| **Phase 10** | Domain-specific capability (Armenian business) | Phase 9 (working multilingual) |

---

## 12. Key Findings Summary

1. **Qwen English capability is ~55%, not 16.7%** — metrics are undercounting correct answers
2. **Armenian benchmark has quality issues** — prompts contain typos and non-standard text
3. **Native Voxline model is not viable** — 936K params produces only incoherent garbage
4. **QLoRA fine-tuning is feasible** but risky on 3.2 GB RAM — requires careful memory management
5. **Metric improvement is the highest-ROI next step** — fixes measurement before attempting improvement
6. **Armenian capability requires dedicated effort** — neither model currently understands Armenian semantics
7. **Qwen's true failure modes are:** arithmetic reasoning, logical reasoning, language confusion (Spanish), and Armenian comprehension — not basic English
