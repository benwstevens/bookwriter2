# Bookwriter2 — Prompt Caching & Cost Estimator Fix

## Status: Complete

## Changes Made

### 1. Prompt Caching (generator.py)
- **`_make_cached_system()`** — New helper that converts a system prompt string (or list) into structured content blocks with `cache_control: {"type": "ephemeral"}`. This enables Anthropic's server-side prompt caching so the system prompt is only transmitted once and reused across subsequent API calls within a 5-minute TTL window.
- **`_api_call_with_retry()`** — Now passes cached system blocks and logs cache read/write token counts.
- **`_stream_api_call_with_retry()`** — Same caching applied; logs cache stats from `stream.get_final_message()`.

### 2. Cost Estimator Fix (generator.py)
- **`_estimate_cost()`** — Rewritten to account for prompt caching pricing:
  - **Cache write**: 1.25x base input price (first call per stage)
  - **Cache read**: 0.1x base input price (subsequent calls)
  - **Uncached input**: 1.0x (user messages that vary per chapter)
  - Separately tracks cached vs uncached tokens for each stage (generation, summary, coherence)
  - Previously treated all input tokens at full price, significantly overestimating cost for multi-chapter books

### Expected Savings
For a 10-chapter book with a ~2000-token system prompt:
- Old estimate charged 20,000 tokens at $5/M = $0.10 for system prompts alone
- New estimate: 2,000 write + 18,000 read = $0.0125 + $0.009 = ~$0.02 (80% savings on system prompt input)
- Savings scale with chapter count — more chapters = more cache hits

## Architecture Notes
- Caching is ephemeral (5-minute TTL) — works well for sequential chapter generation
- System prompt must be ≥1024 tokens for caching to activate (instructions + style guide typically exceeds this)
- No changes needed to callers — the helper handles string-to-block conversion transparently
