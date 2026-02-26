# generator.py

A CLI tool that writes a full book from a YAML table of contents and a plain-text style guide, using Claude as the writing engine. It generates each chapter sequentially, runs a coherence editing pass across chapters, and assembles the final output as both EPUB and HTML.

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/settings/keys) set via the `ANTHROPIC_API_KEY` environment variable or a `.env` file in the project root

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `anthropic`, `ebooklib`, `beautifulsoup4`, `lxml`, `pyyaml`

## Usage

```bash
python3 generator.py <toc.yaml> <style_guide.txt> [options]
```

### Arguments

| Argument | Description |
|---|---|
| `toc.yaml` | Path to a YAML file defining the book's title, author, and chapter list |
| `style_guide.txt` | Path to a plain-text file describing the desired writing style |

### Options

| Flag | Description |
|---|---|
| `--stage N` | Start from stage N (1-5) instead of the beginning. Useful for resuming after a failure. |
| `--dry-run` | Parse inputs and show the plan (cost estimate, chapter breakdown) without making any API calls. |
| `--target N` | Override the per-chapter word target (default: 3500). |
| `--regen N` | Regenerate chapter N by deleting its cached files and re-running stage 2. |

### Examples

```bash
# Full run from scratch
python3 generator.py toc.yaml style_guide.txt

# Preview the plan without spending money
python3 generator.py toc.yaml style_guide.txt --dry-run

# Set each chapter to ~5000 words
python3 generator.py toc.yaml style_guide.txt --target 5000

# Resume from the coherence pass (stages 1-3 already done)
python3 generator.py toc.yaml style_guide.txt --stage 4

# Regenerate only chapter 5
python3 generator.py toc.yaml style_guide.txt --regen 5
```

## Pipeline Stages

### Stage 1 — Parse & Validate Inputs

Loads the YAML table of contents and style guide, validates that all required fields are present, creates the output directory structure, and prints a summary with a cost estimate.

### Stage 2 — Generate Chapters

Generates each chapter sequentially using Claude Opus with extended thinking. Each chapter receives:
- The full table of contents for structural context
- Summaries of all prior chapters (generated via Claude Sonnet)
- The full text of the immediately preceding chapter for tonal continuity
- A description of the next chapter for writing a preview paragraph

Results are cached — re-running skips chapters that already have generated files.

### Stage 3 — Save & Validate

Verifies all chapter files exist and reports word counts against their targets, flagging significant deviations.

### Stage 4 — Coherence Pass

Runs a coherence editing pass over the full manuscript using Claude Opus. For books with more than 7 chapters, it uses overlapping sliding windows (5 chapters each, overlapping by 3) so the model can see enough surrounding context to smooth transitions and fix inconsistencies. Window results are cached.

### Stage 5 — Assemble EPUB + HTML

Reads the coherence-edited chapters and packages them into:
- An `.epub` file with a table of contents and basic CSS styling
- A single combined `.html` file

Both are saved to the book's `output/` directory.

## TOC Format

The table of contents is a YAML file with the following structure:

```yaml
title: "Book Title"
author: "Author Name"
default_word_target: 3500    # optional, defaults to 3500
chapters:
  - title: "Chapter One Title"
    description: >
      A detailed description of what this chapter should cover.
      The more specific, the better the output.
    word_target: 4000        # optional per-chapter override

  - title: "Chapter Two Title"
    description: >
      Description for chapter two.
```

See `sample_toc.yaml` for a working example.

## Output Structure

```
books/<book_slug>/
  toc.yaml                     # copy of input TOC
  style_guide.txt              # copy of input style guide
  generated_chapters/          # stage 2 output
    01 - Chapter_Title_generated.html
    01 - Chapter_Title_summary.txt
    ...
  coherence_edited/            # stage 4 output
    01 - Chapter_Title_edited.html
    ...
  output/                      # stage 5 output
    Book_Title.epub
    Book_Title.html
```
