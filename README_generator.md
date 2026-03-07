# generator.py — Book Generator

A CLI tool that writes a complete, publication-quality non-fiction book from two inputs: a YAML table of contents and a plain-text style guide. It generates each chapter sequentially with Claude Opus, runs a coherence editing pass across chapters, and assembles the final output as both EPUB and HTML.

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-..."

# Always dry-run first to check cost and structure
python3 generator.py my_toc.yaml my_style_guide.txt --dry-run

# Full run
python3 generator.py my_toc.yaml my_style_guide.txt
```

## The Two Inputs You Need to Prepare

The quality of the book depends almost entirely on the quality of these two documents. The script itself is just a pipeline — the TOC and style guide are where the work is.

### 1. Table of Contents (`toc.yaml`)

This is the harder of the two. The TOC is a YAML file with a specific hierarchical structure: **parts → sections → chapters**, where each chapter has detailed internal structure. Getting this right is 80% of the work.

#### Top-Level Fields

```yaml
title: "Your Book Title"
author: "Author Name"
default_word_target: 1300    # words per chapter (can override per-chapter)
```

#### Hierarchy: Parts → Sections → Chapters

```yaml
parts:
  - title: "Part One: Before the First Shovel"
    description: >
      1-3 sentences on what this part covers and why it exists as a unit.

    sections:
      - title: "Why These Projects Exist"
        description: >
          1-2 sentences on this section's focus within the part.

        chapters:
          - title: "Nobody Thinks About It Until It Stops Working"
            word_target: 1300          # optional per-chapter override
            description: >
              2-4 sentences: what this chapter argues, what ground it covers.
```

#### Detailed Chapter Structure (required for every chapter)

This is the part that makes the biggest difference. Every chapter needs these fields:

```yaml
            intro: >
              Guidance for the opening 1-3 paragraphs. Suggest a specific
              opening device — a metaphor from another field, a scene, a
              striking fact. "Open with an analogy from medicine — the
              lymphatic system as the body's most critical and least
              understood system..." is good. "Introduce the topic" is not.

            chapter_sections:          # 3-4 sections per chapter
              - heading: "A Very Brief History of Human Waste"
                description: >
                  A short catchy description of this section's focus.
                bullet_points:         # 3-5 concrete points per section
                  - "Ancient Rome's Cloaca Maxima — engineering without theory"
                  - "The Great Stink of London (1858) as political catalyst"
                  - "Germ theory revolution — science finally catches up"
                  - "Current $271B infrastructure gap (ASCE report card)"

            conclusion: >
              Guidance for closing 1-2 paragraphs. What big idea is now
              clear? What major caveat does everyone in the field know?

            transition: >
              1-2 sentences bridging to the next chapter. Create tension —
              explain how this chapter's information leads into or is better
              understood in context of what comes next. For the final chapter,
              close the book's arc instead.
```

#### TOC Rules

- Every chapter MUST have: `title`, `description`, `intro`, `chapter_sections` (3-4), `conclusion`, `transition`
- Every `chapter_section` MUST have: `heading`, `bullet_points` (3-5 bullets)
- Bullet points should be **concrete** — specific ideas, questions, data points, arguments. Not "discuss the importance of X"
- Chapter titles should be vivid ("Nobody Thinks About It Until It Stops Working") not generic ("Understanding Infrastructure")
- Parts represent genuine shifts in the book's arc (understanding the problem → building the solution → living with the result)
- Use YAML multi-line strings (`>`) for all description, intro, conclusion, and transition fields

#### Generating a TOC with Claude

Writing a good TOC from scratch is the hardest part. Use the included `toc_template.txt` as a prompt — paste it into Claude Chat along with a description of your book's topic, audience, and angle. Claude will generate a properly formatted YAML TOC you can use directly or edit.

```
# In Claude Chat, paste the contents of toc_template.txt and add:
=== BOOK TOPIC ===
A practical guide to developing wastewater treatment plants, written from
the perspective of a real estate developer. Audience is smart generalists.
Conversational tone, first person, lots of real-world war stories.
```

### 2. Style Guide (`style_guide.txt`)

A plain-text file that defines voice, tone, and writing constraints. The generator feeds this to Claude as part of the system prompt for every chapter, so it shapes the entire book. It doesn't need to be long — the sample is 37 lines — but it should be specific.

#### What to Include

Cover these categories (use whatever headings make sense):

| Category | What to specify | Example |
|---|---|---|
| **Voice and tone** | Person, register, who the "author" is | "First person. Conversational but authoritative — like a mentor over coffee." |
| **Target audience** | Who's reading, what they know | "Smart generalists who don't work in the industry." |
| **Reading level** | Sentence/paragraph complexity | "10th-grade reading level. Max 5 sentences per paragraph." |
| **Anecdotes** | Whether to use stories, what kind | "Every chapter needs real-world examples. Flag composites." |
| **Metaphors** | Style of figurative language | "Everyday metaphors. No clichés." |
| **Default word count** | Per-chapter target | "1,500 words per chapter unless otherwise specified." |
| **Humor** | Tone of any humor | "Wry, self-deprecating. Never at anyone's expense." |
| **Structure** | Any structural preferences | "Descriptive subheadings, not generic ones like 'Section 1'." |

#### Full Example (from a successful run)

```
VOICE AND TONE:
Write in first person. The author is a real estate developer with 20+ years of
experience. The tone is conversational but authoritative — like a mentor
explaining the business over coffee. Use "I" freely. Share personal war stories.
Avoid academic jargon but don't talk down to the reader.

TARGET AUDIENCE:
Smart generalists — people who are curious about how buildings get built but
don't work in the industry. Think: a successful professional in another field
who's considering a real estate investment, or a journalist covering a
development controversy. They're intelligent but don't know the terminology.

READING LEVEL:
Aim for a 10th-grade reading level. Short sentences when making key points.
Longer sentences are fine for narrative passages, but break up any paragraph
that runs past 5 sentences.

ANECDOTES:
Every chapter should include real-world stories or examples, always drawn from
real situations. If a metaphor is ever a composite, identify it as such.

METAPHORS:
Use metaphors drawn from everyday life to explain industry concepts. Avoid
clichés ("at the end of the day," "it's a marathon not a sprint").

DEFAULT WORD COUNT:
1,500 words per chapter unless otherwise specified.

HUMOR:
Light humor is encouraged — wry observations about the absurdity of the
process. Never at the expense of any group of people. Self-deprecating
humor about the author's own mistakes is great.

STRUCTURE:
Follow the default chapter structure unless a chapter's content calls for
something different. The body sections should use descriptive subheadings
(not generic ones like "Section 1").
```

## CLI Reference

```bash
python3 generator.py <toc.yaml> <style_guide.txt> [options]
```

| Flag | Description |
|---|---|
| `--dry-run` | Parse inputs and show the plan (cost estimate, chapter breakdown) without making API calls. **Always run this first.** |
| `--stage N` | Start from stage N (1-5). Useful for resuming after a failure or re-running just the coherence pass. |
| `--target N` | Override the per-chapter word target from the command line. |
| `--regen N` | Regenerate chapter N only — deletes its cached files and re-runs stage 2. |

### Common Workflows

```bash
# 1. Check structure and cost before committing
python3 generator.py toc.yaml style_guide.txt --dry-run

# 2. Full run from scratch
python3 generator.py toc.yaml style_guide.txt

# 3. Re-run just the coherence pass (chapters already generated)
python3 generator.py toc.yaml style_guide.txt --stage 4

# 4. Regenerate a chapter you're not happy with, then re-cohere
python3 generator.py toc.yaml style_guide.txt --regen 5
python3 generator.py toc.yaml style_guide.txt --stage 4

# 5. Override word count for all chapters
python3 generator.py toc.yaml style_guide.txt --target 2000
```

## Pipeline Stages

The generator runs 5 stages sequentially. Each stage caches its results, so you can resume from any stage after a failure.

### Stage 1 — Parse & Validate

Loads the YAML TOC and style guide, validates required fields, creates the book directory under `books/`, flattens the hierarchy to a sequential chapter list, prints a summary with per-chapter breakdown, and estimates total API cost. Prompts for confirmation before proceeding.

### Stage 2 — Generate Chapters

Generates each chapter sequentially using **Claude Opus** with adaptive thinking. Each chapter receives rich context:
- The full table of contents (or hierarchy summary for structured books)
- Summaries of all prior chapters (generated cheaply via Claude Sonnet)
- Full text of the immediately preceding chapter (for tonal continuity)
- Description of the next chapter (for writing a natural transition)
- The detailed chapter structure (intro, sections with bullet points, conclusion, transition)

Results cached as HTML in `generated_chapters/`. Re-running skips completed chapters.

### Stage 3 — Save & Validate

Verifies all chapter files exist. Reports actual word counts vs. targets, flagging deviations greater than 10-20%.

### Stage 4 — Coherence Pass

Runs a coherence editing pass using **Claude Opus** with adaptive thinking. For books with more than 7 chapters, uses overlapping sliding windows (5 chapters each, stepping by 2, overlapping by 3) processed concurrently. Edits focus on:
- Smoothing transitions between chapters
- Removing redundant examples or explanations across chapters
- Enforcing consistent terminology
- Fixing cross-references ("as discussed in Chapter 3...")
- Checking the narrative arc
- Maintaining consistent voice and tone

Results cached in `coherence_edited/`.

### Stage 5 — Assemble EPUB + HTML

Packages coherence-edited chapters into:
- An `.epub` with hierarchical table of contents and CSS styling
- A single combined `.html` file

Both saved to the book's `output/` directory.

## Output Structure

```
books/<book-slug>/
├── toc.yaml                              # copy of your input
├── style_guide.txt                       # copy of your input
├── generated_chapters/
│   ├── 01 - Chapter_Title_generated.html
│   ├── 01 - Chapter_Title_summary.txt
│   └── ...
├── coherence_edited/
│   ├── 01 - Chapter_Title_edited.html
│   ├── window_1_5_cache.html             # coherence window cache
│   └── ...
└── output/
    ├── Book_Title.epub
    └── Book_Title.html
```

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/settings/keys) set via `ANTHROPIC_API_KEY` environment variable or `.env` file

```bash
pip install -r requirements.txt
```

Dependencies: `anthropic`, `ebooklib`, `beautifulsoup4`, `lxml`, `pyyaml`, `pymupdf`

## Typical Cost

A 13-chapter book at 1,300 words/chapter runs roughly $5-15 in API costs (Opus for generation and coherence, Sonnet for summaries). The dry-run will give you a precise estimate before you spend anything.

## Reference Files

| File | Purpose |
|---|---|
| `sample_toc.yaml` | Complete working TOC from a successful run — use as a template |
| `sample_style_guide.txt` | Complete working style guide from a successful run |
| `toc_template.txt` | Prompt to paste into Claude Chat to generate a new TOC |
| `generator_instructions.txt` | System prompt used for chapter generation (edit to change writing rules) |
| `generator_coherence_instructions.txt` | System prompt used for coherence pass |
