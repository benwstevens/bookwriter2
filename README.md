# BookWriter

A tool that writes a full non-fiction book using Claude. You give it a table of contents and a style guide; it generates every chapter, runs a coherence editing pass across the manuscript, and outputs a finished EPUB and HTML file.

Total cost per book is roughly $4–8 via the Anthropic API. Runtime is about 30–60 minutes depending on chapter count.

---

## How It Works (The Full Workflow)

There are three steps: plan the book, describe the voice, and run the generator.

### Step 1: Create Your Table of Contents

The table of contents is a YAML file (`toc.yaml`) that tells the generator what to write. It defines every chapter's title, description, internal structure, and how chapters connect to each other. The more specific you are here, the better the output.

You don't write this YAML by hand. You use Claude to generate it from a plain-English description of your book. Here's how:

1. **Open `toc_template.txt` in a text editor.** This file has two halves. The top half (lines 1–61) contains the YAML format rules, structural requirements, and quality constraints. The bottom half (below the `=== BOOK TOPIC ===` line) is a placeholder for your book description.

2. **Delete everything below the `=== BOOK TOPIC ===` line** and replace it with a description of your book. This is where you describe the subject, angle, structure, chapter count, and any specific topics you want covered. Be as detailed or as loose as you want — the template rules will force a thorough, structured output either way. For example:

   ```
   === BOOK TOPIC ===
   A book about developing a wastewater treatment plant, told from the
   perspective of someone who's done 10 of these and knows what really
   happens behind the scenes.

   Part 1: history, politics, funding — why these projects happen
   Part 2: the design process — consultants, engineers, how it works
   Part 3: construction — what actually goes wrong and how you manage it
   Part 4: commissioning and lessons learned

   13 chapters, 1200-1500 words each.

   I want a chapter specifically about how funding works — the alphabet
   soup of SRF, USDA, WIFIA, and how communities that need the most
   help have the least capacity to navigate the process.
   ```

3. **Copy the entire file** (template rules + your book description) **and paste it into [Claude](https://claude.ai) as a single message.** Claude reads the formatting rules from the top half and the book concept from the bottom half, and outputs the complete `toc.yaml` directly.

4. **Iterate in that same chat if needed.** Ask Claude to revise: "Make chapter 5 more focused on hydraulics." "Split chapter 7 into two." "The transition from Part 2 to Part 3 feels abrupt." Claude revises the YAML each time.

5. **When you're happy with the result, save it as `toc.yaml`.**

The key idea: the template is a *generation tool*, not a *conversion tool*. The richness of the output — concrete bullet points, specific opening devices, tension in the transitions — comes from the rules baked into the template. You don't need to ask for that level of detail in your book description; the template demands it automatically.

#### What the TOC looks like

The generated YAML has a hierarchy: **parts → sections → chapters**. Each chapter includes:

- `description`: What the chapter argues and covers
- `intro`: Guidance for the opening paragraphs (a specific scene, metaphor, or hook)
- `chapter_sections`: 3–4 internal sections, each with a heading and 3–5 concrete bullet points
- `conclusion`: Guidance for wrapping up the chapter's argument
- `transition`: 1–2 sentences bridging to the next chapter

See `sample_toc.yaml` for a full working example.

### Step 2: Write a Style Guide

The style guide is a plain-text file (`style_guide.txt`) that describes the voice and tone you want. It gets included in every API call, so the generator writes consistently across all chapters.

At minimum, cover:

- **Voice and tone** — First person? Third person? Conversational or formal?
- **Target audience** — Who is this for? What do they already know?
- **Reading level** — Short punchy sentences, or long flowing prose?
- **Anecdotes** — Should chapters include stories and examples?
- **Humor** — What kind, if any?

See `sample_style_guide.txt` for an example.

### Step 3: Run the Generator

```bash
# Install dependencies (once)
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Preview the plan without spending money
python3 generator.py toc.yaml style_guide.txt --dry-run

# Full run
python3 generator.py toc.yaml style_guide.txt
```

The generator runs a 5-stage pipeline:

| Stage | What it does |
|---|---|
| **1. Parse & Validate** | Loads your TOC and style guide, creates the output directory, prints a cost estimate |
| **2. Generate Chapters** | Writes each chapter sequentially using Claude Opus. Each chapter receives the full TOC, the preceding chapter's text, and summaries of all earlier chapters for continuity |
| **3. Save & Validate** | Verifies all chapters exist and reports word counts vs. targets |
| **4. Coherence Pass** | Runs an editing pass over the full manuscript using overlapping windows, smoothing transitions, removing redundancy, and fixing inconsistencies |
| **5. Assemble Output** | Packages the edited chapters into an EPUB file and a combined HTML file |

Everything is cached. If the generator fails or you stop it mid-run, re-running the same command skips completed work and picks up where it left off.

---

## CLI Reference

```bash
python3 generator.py <toc.yaml> <style_guide.txt> [options]
```

| Flag | Description |
|---|---|
| `--dry-run` | Parse inputs and show the plan (cost estimate, chapter breakdown) without making API calls |
| `--stage N` | Start from stage N (1–5). Useful for resuming or re-running later stages after manual edits |
| `--target N` | Override the per-chapter word target (default comes from your TOC) |
| `--regen N` | Regenerate chapter N only (deletes its cached files and re-runs stage 2) |

### Examples

```bash
# Set each chapter to ~5000 words
python3 generator.py toc.yaml style_guide.txt --target 5000

# Resume from the coherence pass (stages 1-3 already done)
python3 generator.py toc.yaml style_guide.txt --stage 4

# Regenerate only chapter 5 (maybe you revised its TOC entry)
python3 generator.py toc.yaml style_guide.txt --regen 5
```

---

## Output Structure

After a full run, your book lives in `books/<book_slug>/`:

```
books/developing_a_wastewater_treatment_plant/
  toc.yaml                          # Copy of your input TOC
  style_guide.txt                   # Copy of your input style guide
  generated_chapters/               # Stage 2 output
    01 - Chapter_Title_generated.html
    01 - Chapter_Title_summary.txt
    ...
  coherence_edited/                 # Stage 4 output
    01 - Chapter_Title_edited.html
    ...
  output/                           # Final output
    Book_Title.epub
    Book_Title.html
```

---

## Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/settings/keys) (set via `ANTHROPIC_API_KEY` environment variable or a `.env` file in the project root)

Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `anthropic`, `ebooklib`, `beautifulsoup4`, `lxml`, `pyyaml`, `markdown`

---

## Files in This Repo

| File | What it is |
|---|---|
| `generator.py` | The main pipeline script (all 5 stages) |
| `toc_template.txt` | Template you paste into Claude Chat to generate a `toc.yaml` (see Step 1 above) |
| `sample_toc.yaml` | Example TOC for a book about real estate development |
| `sample_style_guide.txt` | Example style guide |
| `generator_instructions.txt` | System prompt sent to Claude during chapter generation (you don't need to edit this) |
| `generator_coherence_instructions.txt` | System prompt sent to Claude during the coherence pass (you don't need to edit this) |
| `shared.py` | Utility functions used by the generator |
| `requirements.txt` | Python dependencies |

---

## Cost and Runtime

For a 13-chapter book at ~1500 words per chapter:

- **API cost:** ~$4–6
- **Runtime:** ~30–50 minutes (generation ~10 min, coherence ~10–25 min)
- **Models used:** Claude Opus 4.6 (generation + coherence), Claude Sonnet 4.6 (chapter summaries)

The `--dry-run` flag shows a cost estimate before you commit to a full run.

---

## Known Quirks

- **Chapters tend to run 20–30% over word targets.** Opus is verbose. If you want 1500-word chapters, consider setting your target to 1200.
- **Coherence pass log output can interleave** when multiple windows run concurrently. The results are correct; the progress messages just overlap.
