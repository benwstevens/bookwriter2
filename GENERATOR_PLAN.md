# Book Generator — Project Plan

## What This Is

A new script (`generator.py`) that reverses the distiller pipeline: instead of taking a book and condensing it, it takes a table of contents + style guide and **writes** a full book, chapter by chapter, with a coherence pass and EPUB assembly at the end.

---

## Handoff Protocol

**When context runs long:** If I (Claude) start running low on context window space during implementation, I will notify you immediately and provide a handoff prompt you can paste into a new conversation window. The handoff prompt will include: what files have been created so far, what stage of implementation we're at, what remains to be done, and any design decisions made during implementation that a fresh instance would need to know.

---

## Inputs

The user provides two files:

### 1. Table of Contents (`toc.yaml`)

YAML format. Each chapter entry includes a title and a brief description of what the chapter should cover. Optional per-chapter word target overrides.

```yaml
title: "The Birth of a Building"
author: "Ben Stevens"
chapters:
  - title: "Why Buildings Take So Long"
    description: >
      Introduce the reader to the development timeline.
      Cover the gap between public perception and reality.
      Use the metaphor of pregnancy — a building gestates.
    word_target: 4000  # optional override

  - title: "Dirt, Dollars, and Due Diligence"
    description: >
      The land acquisition process. Due diligence, environmental,
      zoning, entitlements. Why most deals die here.
```

### 2. Style Guide (`style_guide.txt`)

A plain-text file that defines the voice, structure, and constraints for the writing. This gets sent as part of the system prompt on every API call. The default chapter structure (unless the style guide says otherwise) is:

- **Opening (2–3 paragraphs):** Establish the topic's relevance with a catchy metaphor or anecdote
- **Body (3 sections × 3–4 paragraphs each):** The substantive content, with section subheadings
- **Conclusion (1–2 paragraphs):** Wrap up the chapter's argument
- **Preview (1 short paragraph):** "In the next chapter" teaser bridging to what comes next

The style guide can specify things like: target audience, reading level, tone, whether to use first person, whether to include anecdotes, default word count per chapter, etc.

---

## Pipeline Stages

Modeled after the distiller's stage system, with `--stage N` to re-run from any point.

### Stage 1: Parse & Validate Inputs
- Load `toc.yaml`, validate structure
- Load `style_guide.txt`
- Create the book directory under `books/<slug>/`
- Create subdirectories: `generated_chapters/`, `coherence_edited/`, `output/`
- Compute per-chapter word targets (global default from style guide, with per-chapter overrides from TOC)
- Print a summary: chapter count, word targets, cost estimate
- Prompt user to confirm before proceeding

### Stage 2: Generate Chapters (the big one)
- Process chapters sequentially (order matters — each chapter gets context from prior ones)
- For each chapter, the API call includes:
  - **System prompt:** The style guide + the default chapter structure template + HTML formatting rules
  - **User message:** The chapter's title and description from the TOC, the word target, and *context*
- **Context strategy:**
  - Chapter 1: Just the full TOC (so the model knows the arc of the book)
  - Chapter 2+: Full TOC + full text of the immediately preceding chapter + one-paragraph summaries of all earlier chapters
  - The one-paragraph summaries are generated as a side-effect: after each chapter is written, a quick cheap API call (Sonnet, no extended thinking) produces a ~100-word summary that gets saved alongside the chapter
- **Model:** `claude-opus-4-6` with extended thinking enabled (budget_tokens: 10000)
- **Caching:** Skip chapters whose output file already exists (same pattern as distiller — lets you re-run after a failure without re-generating completed chapters)
- **Retry logic:** Same exponential backoff pattern as distiller (5 retries, 2^n seconds)
- Save each chapter as `01 - Chapter_Title_generated.html` in `generated_chapters/`
- Save each summary as `01 - Chapter_Title_summary.txt` alongside it

### Stage 3: Save & Validate
- Verify all chapter files exist
- Print word counts per chapter and total
- Flag any chapters significantly over/under target

### Stage 4: Coherence Pass
- Read all generated chapters
- **Chunking strategy** (because a full book won't fit in one API call):
  - Process in overlapping windows of 5 chapters: [1–5], [3–7], [5–9], etc.
  - Each window call gets: the full TOC (for arc awareness), the style guide, the full text of the chapters in that window, and coherence instructions
  - The coherence instructions tell the model to focus edits on the *middle* chapters of each window (the ones that overlap with both the prior and next window), and to leave the edges lighter-touch since they'll be covered by adjacent windows
  - Chapter 1 and the final chapter get special treatment as they're only in one window each
  - For short books (≤7 chapters), send everything in a single call
- **What the coherence pass does:**
  - Smooth transitions between chapters — the "in the next chapter" preview in Ch. N should actually match what Ch. N+1 delivers
  - Remove redundancy where multiple chapters repeat the same point
  - Ensure consistent terminology, metaphors, and examples across chapters
  - Verify the "arc" works — early chapters set up concepts that later chapters build on
  - Make sure cross-references ("as we discussed in Chapter 3...") are accurate
- **Model:** `claude-opus-4-6` with extended thinking, streaming enabled (these will be large responses)
- Save coherence-edited chapters to `coherence_edited/`

### Stage 5: Assemble EPUB + HTML
- Reuse the EPUB assembly logic from `distiller.py` stage 5 (nearly identical)
- Build EPUB with proper TOC, CSS styling, chapter structure
- Also output a single combined HTML file
- Save to `output/`

---

## File Structure (Runtime)

```
books/
  birth_of_a_building/
    toc.yaml                          # copied from input
    style_guide.txt                   # copied from input
    generated_chapters/
      01 - Why_Buildings_Take_So_Long_generated.html
      01 - Why_Buildings_Take_So_Long_summary.txt
      02 - Dirt_Dollars_and_Due_Diligence_generated.html
      02 - Dirt_Dollars_and_Due_Diligence_summary.txt
      ...
    coherence_edited/
      01 - Why_Buildings_Take_So_Long_edited.html
      02 - Dirt_Dollars_and_Due_Diligence_edited.html
      ...
    output/
      Birth_of_a_Building.epub
      Birth_of_a_Building.html
```

---

## Files to Create

| File | Purpose |
|---|---|
| `generator.py` | Main script — all 5 stages |
| `generator_instructions.txt` | System prompt for chapter generation (style template, HTML formatting rules, chapter structure) |
| `generator_coherence_instructions.txt` | System prompt for the coherence pass |

We also need minor additions to `shared.py`:
- Add `"generated_chapters_dir"`, `"generated_summaries_dir"`, `"coherence_edited_dir"` to `setup_book_dir()` — or, more likely, the generator handles its own directory setup since its inputs are different (no EPUB source)
- Reuse: `get_api_key()`, `markdown_to_html()`, `wrap_response_html()`, `count_words()`, `estimate_tokens()`, EPUB assembly logic from stage 5

---

## CLI Interface

```bash
# Full run from scratch
python3 generator.py toc.yaml style_guide.txt

# Re-run from a specific stage
python3 generator.py toc.yaml style_guide.txt --stage 4

# Dry run — parse inputs, show plan, estimate cost, but don't call API
python3 generator.py toc.yaml style_guide.txt --dry-run

# Override default word target per chapter
python3 generator.py toc.yaml style_guide.txt --target 3500

# Regenerate a specific chapter (delete its cached file and re-run stage 2)
python3 generator.py toc.yaml style_guide.txt --regen 5
```

---

## Cost Estimate Logic

**Opus pricing:** $15/M input tokens, $75/M output tokens. Extended thinking tokens billed as output.
**Sonnet pricing:** $3/M input tokens, $15/M output tokens.

Per chapter (Opus with extended thinking):
- Input: ~2,000 tokens (system) + ~500 tokens (TOC) + ~4,000 tokens (prior chapter) + ~1,000 tokens (summaries) ≈ ~7,500 input tokens
- Thinking: ~10,000 tokens (budget_tokens)
- Output: ~5,000 tokens (~3,500 words)
- Per-chapter cost: ~(7,500 * $15/M) + ~(15,000 * $75/M) ≈ ~$0.11 + ~$1.13 ≈ ~$1.24

Per summary (Sonnet, no extended thinking):
- Input: ~6,000 tokens, Output: ~150 tokens
- Per-summary cost: ~$0.02 + ~$0.002 ≈ ~$0.02

For a 15-chapter book:
- Generation pass (Opus): ~$18.60
- Summary generation (Sonnet): ~$0.30
- Coherence pass (Opus, ~4 window calls): ~$8–15
- **Total estimate: ~$27–34**

The script will compute and display this before prompting the user to proceed.

---

## Implementation Order

1. **`generator_instructions.txt`** — Get the chapter-writing system prompt right first. This is the creative core.
2. **`generator_coherence_instructions.txt`** — The coherence pass prompt.
3. **`generator.py`** stages 1–3 — Input parsing, chapter generation loop, save/validate.
4. **`generator.py`** stages 4–5 — Coherence pass with chunking, EPUB assembly.
5. **Test with a small TOC** (3 chapters) to validate the full pipeline end-to-end.

---

## Design Decisions Log

- **Opus over Sonnet** for generation quality. Extended thinking enabled by default (budget_tokens: 10000). Both are API costs, not monthly plan usage.
- **No human-in-the-loop** between generation and coherence — runs straight through. User can always re-run from stage 4 after manual edits.
- **Non-fiction only** for v1. Fiction mode can be added later with a separate instruction file.
- **YAML for TOC** because it's more readable than JSON for this use case and handles multi-line descriptions cleanly.
- **Sequential generation** (not parallel) because each chapter needs context from prior ones. This is slower but produces much better continuity.
- **Overlapping coherence windows** rather than one giant call, because a 70k-word book would blow the context window. The overlap ensures no chapter-to-chapter transition falls in a gap.

---

## Codebase Review Notes (from bookprocessing-main)

After reviewing the existing bookprocessing codebase, here are the key findings that affect implementation:

### Reusable from `shared.py` (copied into repo)
- `get_api_key()` — reads from env var or .env file, works as-is
- `markdown_to_html()` — converts Markdown to HTML if detected, useful as fallback
- `wrap_response_html()` — wraps API response in full HTML doc, works as-is
- `count_words()` — strips HTML tags, counts whitespace-split tokens
- `estimate_tokens()` — `len(text) // 4` approximation
- `sanitize_filename()` — cleans text for filenames (max 60 chars)

### NOT reusable (generator needs its own)
- `setup_book_dir()` — designed for EPUB-source pipelines; generator needs custom dir setup (no source_dir, no chapters_dir, different subdirectories)
- `resolve_epub_path()` — not applicable (generator takes YAML + TXT, not EPUB)
- `detect_chapter_tag()`, `split_chapters()`, `filter_chapters()` — only for existing books
- `detect_book_metadata()` — reads from HTML; generator gets metadata from YAML

### EPUB Assembly Pattern (from distiller stage 5)
The distiller's EPUB assembly:
1. Creates `EpubBook()` with metadata (title, author, language)
2. Adds CSS stylesheet: serif fonts, 1.6 line-height, justified text
3. Splits final HTML on `<h2>` tags to create chapters
4. Each chapter gets `file_name=chapter_{i:02d}.xhtml` with CSS link
5. Builds TOC with `epub.Link()` entries
6. Adds NCX + Nav navigation
7. Sets spine: `["nav"] + epub_chapters`
8. Writes with `epub.write_epub()`

This pattern adapts directly — the generator's coherence-edited chapters already have `<h2>` titles.

### Retry Pattern (from distiller)
```python
max_retries = 5; backoff = 2
for attempt in range(max_retries):
    try:
        message = client.messages.create(...)
        break
    except (anthropic.RateLimitError, anthropic.APIError):
        wait = backoff * (2 ** attempt)  # 2, 4, 8, 16, 32
        time.sleep(wait)
```

### Streaming Pattern (from distiller coherence pass)
Distiller uses `client.messages.stream()` for the coherence pass. Generator should do the same for coherence windows.

### Model IDs
- Generation: `claude-opus-4-6` (extended thinking, budget_tokens: 10000)
- Summaries: `claude-sonnet-4-5-20250514` (cheap, no extended thinking)
- Coherence: `claude-opus-4-6` (extended thinking, streaming)

### Cost Pricing (Opus)
- Opus input: $15/M tokens, output: $75/M tokens
- Sonnet input: $3/M tokens, output: $15/M tokens
- Extended thinking tokens billed as output

---

## Implementation Progress

- [x] Review GENERATOR_PLAN.md and bookprocessing codebase
- [x] Copy shared.py and requirements.txt into repo
- [ ] Create `generator_instructions.txt`
- [ ] Create `generator_coherence_instructions.txt`
- [ ] Implement `generator.py` Stage 1 (Parse & Validate)
- [ ] Implement `generator.py` Stage 2 (Generate Chapters)
- [ ] Implement `generator.py` Stage 3 (Save & Validate)
- [ ] Implement `generator.py` Stage 4 (Coherence Pass)
- [ ] Implement `generator.py` Stage 5 (EPUB + HTML Assembly)
- [ ] Create sample `toc.yaml` for testing
- [ ] End-to-end dry-run test
