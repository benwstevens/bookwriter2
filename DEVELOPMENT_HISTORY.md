# Development History

## Session 1 — Feb 26, 2026: Initial Build

Built the entire generator pipeline from scratch. The project started from an existing `bookprocessing` codebase (a book *distiller* that condenses existing books) and created the reverse: a book *generator* that writes a full book from a table of contents and style guide.

### What was built
- **`generator.py`** (~800 lines): Complete 5-stage pipeline
  - Stage 1: Parse & validate YAML TOC + style guide, create directory structure, estimate cost
  - Stage 2: Generate chapters sequentially with Opus, each chapter receiving context from prior chapters (full preceding chapter + summaries of earlier ones)
  - Stage 3: Save & validate — verify all chapters exist, check word counts against targets
  - Stage 4: Coherence pass — overlapping windows of 5 chapters processed through Opus to smooth transitions, remove redundancy, and ensure consistency
  - Stage 5: Assemble final EPUB + combined HTML
- **`generator_instructions.txt`**: System prompt for chapter generation (HTML formatting rules, chapter structure template)
- **`generator_coherence_instructions.txt`**: System prompt for the coherence editing pass
- **`shared.py`**: Ported utility functions from the bookprocessing codebase (`get_api_key`, `markdown_to_html`, `count_words`, `estimate_tokens`, `sanitize_filename`, etc.)
- **`sample_toc.yaml`** and **`sample_style_guide.txt`**: Example inputs for testing
- **`GENERATOR_PLAN.md`**: Detailed project plan and design decisions
- **`requirements.txt`**: Dependencies (anthropic, ebooklib, beautifulsoup4, lxml, pyyaml, markdown)

### Key design decisions
- Opus for generation quality, Sonnet for cheap summary generation
- Sequential chapter generation (not parallel) because each chapter needs prior context
- Overlapping coherence windows (step by 2, window size 5) so no chapter transition falls in a gap
- Extended thinking enabled for both generation and coherence
- Full caching at every stage — can resume after failures without re-generating completed work

---

## Session 2 — Feb 28, 2026: Hierarchical TOC, API Fixes, Performance

### 1. Hierarchical TOC Support (major feature)
**Commit `7ba4a01`** — The original TOC format was a flat list of chapters. Added full hierarchical support: **parts → sections → chapters**. This is the natural structure for non-fiction books.

Changes:
- New TOC format supports nested `parts` containing `sections` containing `chapters`
- Each chapter can now specify: `intro` guidance, detailed `chapter_sections` with bullet points, `conclusion` guidance, and `transition` text
- `_flatten_chapters()` function converts hierarchical TOC to a flat chapter list with part/section metadata preserved on each chapter
- `_build_hierarchy_summary()` generates a readable outline for inclusion in API prompts
- Stage 1 prints the full part/section/chapter tree
- Stage 2 generation prompts include hierarchy context and structural position
- Stage 4 coherence pass includes part/section boundary notes so the model pays attention to structural transitions
- Stage 5 EPUB assembly creates proper part title pages between sections
- Full backward compatibility with flat TOC files (no parts → works exactly as before)
- Updated `sample_toc.yaml` with a rich hierarchical example
- Updated `generator_instructions.txt` and `generator_coherence_instructions.txt` for hierarchy awareness
- Also added `toc_template.txt` — a template for prompting Claude Chat to generate well-structured TOCs

### 2. Fix: Adaptive Thinking API Migration
**Commits `9a809dd`, `3c26580`** — The Anthropic API deprecated `thinking.type: "enabled"` with `budget_tokens` in favor of `thinking.type: "adaptive"`. The initial code used the old format, which broke on first run attempt.

- First fix: switched `type` from `"enabled"` to `"adaptive"`
- Second fix: removed `budget_tokens` parameter entirely — adaptive mode doesn't accept it (Claude determines its own thinking budget based on problem complexity)

### 3. Fix: Invalid Summary Model ID
**Commit `8679ae9`** — The summary generation model was set to `claude-sonnet-4-5-20250514`, which doesn't exist (the correct Sonnet 4.5 snapshot date is `20250929`). Upgraded to `claude-sonnet-4-6` — same price ($3/$15 per million tokens), newer model.

### 4. Parallelize Coherence Pass
**Commit `04ee01d`** — The coherence pass ran 5 overlapping windows sequentially, taking ~25 minutes for a 13-chapter book. Since windows don't depend on each other's output (overlap resolution is purely post-hoc), all windows can run concurrently.

- Extracted per-window logic into a `_process_window()` helper
- All windows now fire simultaneously via `concurrent.futures.ThreadPoolExecutor`
- Results collected and overlap resolution applied in deterministic window order afterward
- Expected to cut coherence time from ~25 min to ~10 min (limited by the slowest single window)

### 5. Fix: Cost Estimate Was 3x Too High
**Commit `2a96faa`** — The cost estimator used legacy Opus 4.0/4.1 pricing ($15/$75 per million tokens). Opus 4.6 is $5/$25 — 3x cheaper. Also reduced the assumed thinking token budget from 10K to 5K per call to better reflect adaptive thinking behavior.

Result: estimate dropped from ~$26 to a more accurate ~$5-6 for a 13-chapter book. Actual API bill for the first full run was ~$4.

### First Successful Full Run
Generated **"Developing a Wastewater Treatment Plant"** — a 13-chapter, 4-part non-fiction book:
- 13 chapters generated successfully (~22,600 words total, targets were ~17,400)
- All chapters ran 20-39% over word targets (known issue, not yet addressed)
- 5 coherence windows completed
- Final EPUB and HTML assembled
- Total API cost: ~$4
- Total runtime: ~50 minutes (generation ~10 min, coherence ~25 min sequential, assembly instant)

---

## Current Architecture

```
bookwriter2/
├── generator.py                         # Main pipeline (1,265 lines, 5 stages)
├── shared.py                            # Shared utilities (592 lines)
├── generator_instructions.txt           # System prompt for chapter generation
├── generator_coherence_instructions.txt # System prompt for coherence pass
├── requirements.txt                     # Python dependencies
├── GENERATOR_PLAN.md                    # Design doc and implementation plan
├── sample_toc.yaml                      # Example hierarchical TOC
├── sample_style_guide.txt              # Example style guide
├── toc_template.txt                     # Template for generating TOCs via Claude Chat
├── README_generator.md                  # Usage documentation
└── books/                               # Generated books (gitignored)
    └── <book_slug>/
        ├── generated_chapters/          # Raw chapter HTML + summaries
        ├── coherence_edited/            # Post-coherence chapters + window caches
        └── output/                      # Final EPUB + HTML
```

### Models Used
- **Generation & Coherence:** `claude-opus-4-6` with adaptive thinking
- **Summaries:** `claude-sonnet-4-6` (no thinking, cheap)

### Key CLI Commands
```bash
python3 generator.py toc.yaml style_guide.txt              # Full run
python3 generator.py toc.yaml style_guide.txt --stage 4    # Resume from stage 4
python3 generator.py toc.yaml style_guide.txt --dry-run    # Parse + estimate only
python3 generator.py toc.yaml style_guide.txt --regen 5    # Regenerate chapter 5
python3 generator.py toc.yaml style_guide.txt --target 3500 # Override word target
```

---

## Known Issues / Future Work

- **Word counts consistently overshoot targets by 20-39%.** The generation prompt may need stronger constraints, or the targets should be set lower to compensate for Opus's verbosity.
- **Cost estimate in GENERATOR_PLAN.md is stale.** The detailed per-chapter math in the plan doc still references old pricing and token assumptions. The code's `_estimate_cost()` is now correct.
- **Coherence pass progress output interleaves** when running concurrent windows. Cosmetic only — results are correct.
