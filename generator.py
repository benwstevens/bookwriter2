#!/usr/bin/env python3
"""
Book Generator — Write a full book from a table of contents and style guide.

Takes a YAML table of contents and a plain-text style guide, then generates
a complete book chapter by chapter using Claude, with a coherence pass and
EPUB assembly at the end.

Usage:
    python3 generator.py toc.yaml style_guide.txt
    python3 generator.py toc.yaml style_guide.txt --stage 4
    python3 generator.py toc.yaml style_guide.txt --dry-run
    python3 generator.py toc.yaml style_guide.txt --target 3500
    python3 generator.py toc.yaml style_guide.txt --regen 5
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import yaml
from bs4 import BeautifulSoup

from shared import (
    BASE_DIR,
    count_words,
    estimate_tokens,
    get_api_key,
    markdown_to_html,
    sanitize_filename,
)

INSTRUCTIONS_FILE = BASE_DIR / "generator_instructions.txt"
COHERENCE_INSTRUCTIONS_FILE = BASE_DIR / "generator_coherence_instructions.txt"

DEFAULT_WORD_TARGET = 3500


# ===========================================================================
# Directory setup (generator-specific — different from distiller)
# ===========================================================================
def setup_generator_dirs(toc_data: dict) -> dict[str, Path]:
    """Create the book directory structure for the generator pipeline."""
    title = toc_data.get("title", "Untitled")
    slug = re.sub(r"[^\w\-]", "", title.replace(" ", "_")).lower()
    book_dir = BASE_DIR / "books" / slug

    paths = {
        "book_dir": book_dir,
        "generated_chapters_dir": book_dir / "generated_chapters",
        "coherence_edited_dir": book_dir / "coherence_edited",
        "output_dir": book_dir / "output",
    }

    for _key, p in paths.items():
        p.mkdir(parents=True, exist_ok=True)

    return paths


# ===========================================================================
# Stage 1: Parse & Validate Inputs
# ===========================================================================
def stage1(toc_path: Path, style_path: Path, target_override: int | None = None):
    """Parse and validate inputs. Returns (toc_data, style_guide, paths, chapter_targets)."""
    print("\n" + "=" * 60)
    print("STAGE 1: Parse & Validate Inputs")
    print("=" * 60)

    # Load and validate TOC
    if not toc_path.exists():
        print(f"ERROR: TOC file not found: {toc_path}")
        sys.exit(1)

    with open(toc_path, "r", encoding="utf-8") as f:
        toc_data = yaml.safe_load(f)

    if not toc_data or "chapters" not in toc_data:
        print("ERROR: TOC must have a 'chapters' list.")
        sys.exit(1)

    if not toc_data.get("title"):
        print("ERROR: TOC must have a 'title' field.")
        sys.exit(1)

    chapters = toc_data["chapters"]
    if not chapters or not isinstance(chapters, list):
        print("ERROR: 'chapters' must be a non-empty list.")
        sys.exit(1)

    for i, ch in enumerate(chapters, 1):
        if not ch.get("title"):
            print(f"ERROR: Chapter {i} missing 'title'.")
            sys.exit(1)
        if not ch.get("description"):
            print(f"ERROR: Chapter {i} ('{ch['title']}') missing 'description'.")
            sys.exit(1)

    # Load style guide
    if not style_path.exists():
        print(f"ERROR: Style guide not found: {style_path}")
        sys.exit(1)

    style_guide = style_path.read_text(encoding="utf-8").strip()
    if not style_guide:
        print("ERROR: Style guide is empty.")
        sys.exit(1)

    # Set up directories
    paths = setup_generator_dirs(toc_data)

    # Copy inputs into book directory for reproducibility
    dest_toc = paths["book_dir"] / "toc.yaml"
    dest_style = paths["book_dir"] / "style_guide.txt"
    if not dest_toc.exists():
        shutil.copy2(toc_path, dest_toc)
    if not dest_style.exists():
        shutil.copy2(style_path, dest_style)

    # Compute per-chapter word targets
    global_target = target_override or toc_data.get("default_word_target", DEFAULT_WORD_TARGET)
    chapter_targets = []
    for ch in chapters:
        wt = ch.get("word_target", global_target)
        chapter_targets.append({
            "title": ch["title"],
            "description": ch["description"].strip(),
            "word_target": wt,
        })

    # Print summary
    total_target = sum(ct["word_target"] for ct in chapter_targets)
    print(f"\n  Book: {toc_data['title']}")
    print(f"  Author: {toc_data.get('author', 'Unknown')}")
    print(f"  Chapters: {len(chapter_targets)}")
    print(f"  Default word target: {global_target:,} per chapter")
    print(f"  Total target: ~{total_target:,} words")

    print(f"\n  Per-chapter breakdown:")
    for i, ct in enumerate(chapter_targets, 1):
        print(f"    {i:2d}. {ct['title']} — {ct['word_target']:,} words")

    # Cost estimate
    cost_estimate = _estimate_cost(chapter_targets, style_guide, toc_data)
    print(f"\n  Estimated cost: ~${cost_estimate:.2f}")

    return toc_data, style_guide, paths, chapter_targets


def _estimate_cost(chapter_targets, style_guide, toc_data):
    """Estimate total API cost for the full pipeline."""
    instructions_tokens = estimate_tokens(
        INSTRUCTIONS_FILE.read_text(encoding="utf-8") if INSTRUCTIONS_FILE.exists() else ""
    )
    style_tokens = estimate_tokens(style_guide)
    toc_tokens = estimate_tokens(yaml.dump(toc_data))

    # Generation pass (Opus): per chapter
    gen_input_total = 0
    gen_output_total = 0
    summary_context_tokens = 0

    for i, ct in enumerate(chapter_targets):
        # System: instructions + style guide
        sys_tokens = instructions_tokens + style_tokens
        # User: TOC + description + prior chapter (~word_target * 1.3 tokens) + summaries
        prior_chapter_tokens = int(chapter_targets[i - 1]["word_target"] * 1.3) if i > 0 else 0
        user_tokens = toc_tokens + estimate_tokens(ct["description"]) + prior_chapter_tokens + summary_context_tokens

        gen_input_total += sys_tokens + user_tokens
        gen_output_total += int(ct["word_target"] * 1.3)  # tokens ≈ words * 1.3

        # Accumulate summary tokens for next iteration (~150 tokens per summary)
        summary_context_tokens += 150

    # Extended thinking tokens (billed as output): 10000 per chapter
    thinking_tokens = 10000 * len(chapter_targets)

    # Summary generation (Sonnet): ~6000 input + ~150 output per chapter
    summary_input_total = 6000 * len(chapter_targets)
    summary_output_total = 150 * len(chapter_targets)

    # Coherence pass (Opus): rough estimate
    num_windows = max(1, (len(chapter_targets) - 3) // 2 + 1) if len(chapter_targets) > 7 else 1
    coherence_input_per_window = 5 * int(sum(ct["word_target"] for ct in chapter_targets) / len(chapter_targets) * 1.3)
    coherence_input_total = coherence_input_per_window * num_windows + toc_tokens * num_windows + style_tokens * num_windows
    coherence_output_total = coherence_input_per_window * num_windows  # roughly same size
    coherence_thinking = 10000 * num_windows

    # Opus: $15/M input, $75/M output
    opus_input_cost = (gen_input_total + coherence_input_total) / 1_000_000 * 15.0
    opus_output_cost = (gen_output_total + thinking_tokens + coherence_output_total + coherence_thinking) / 1_000_000 * 75.0

    # Sonnet: $3/M input, $15/M output
    sonnet_input_cost = summary_input_total / 1_000_000 * 3.0
    sonnet_output_cost = summary_output_total / 1_000_000 * 15.0

    return opus_input_cost + opus_output_cost + sonnet_input_cost + sonnet_output_cost


# ===========================================================================
# Stage 2: Generate Chapters
# ===========================================================================
def stage2(toc_data, style_guide, paths, chapter_targets, regen_chapter=None):
    """Generate each chapter sequentially, with context from prior chapters."""
    print("\n" + "=" * 60)
    print("STAGE 2: Generate Chapters")
    print("=" * 60)

    api_key = get_api_key()

    if not INSTRUCTIONS_FILE.exists():
        print(f"ERROR: {INSTRUCTIONS_FILE} not found.")
        sys.exit(1)
    instructions = INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()

    # Build system prompt: instructions + style guide
    system_prompt = instructions + "\n\n--- STYLE GUIDE ---\n\n" + style_guide

    # Full TOC as YAML for context
    toc_yaml = yaml.dump(toc_data, default_flow_style=False, allow_unicode=True)

    gen_dir = paths["generated_chapters_dir"]

    # If --regen specified, delete that chapter's cached files
    if regen_chapter is not None:
        idx = regen_chapter
        for ct in [chapter_targets[idx - 1]] if 1 <= idx <= len(chapter_targets) else []:
            safe_title = sanitize_filename(ct["title"])
            pattern = f"{idx:02d} - {safe_title}"
            for f in gen_dir.glob(f"{pattern}*"):
                print(f"  Removing cached file: {f.name}")
                f.unlink()

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    gen_model = "claude-opus-4-6"
    summary_model = "claude-sonnet-4-5-20250514"

    summaries = []  # list of summary strings for context
    failures = []

    for i, ct in enumerate(chapter_targets, start=1):
        safe_title = sanitize_filename(ct["title"])
        chapter_filename = f"{i:02d} - {safe_title}_generated.html"
        summary_filename = f"{i:02d} - {safe_title}_summary.txt"
        chapter_path = gen_dir / chapter_filename
        summary_path = gen_dir / summary_filename

        print(
            f"\nChapter {i}/{len(chapter_targets)}: {ct['title']} "
            f"(~{ct['word_target']} words)…",
        )

        # Check cache
        if chapter_path.exists() and summary_path.exists():
            print("  Already generated, skipping.")
            summaries.append(summary_path.read_text(encoding="utf-8").strip())
            continue

        # Build user message with context
        context_parts = [f"BOOK TABLE OF CONTENTS:\n{toc_yaml}"]

        # Add summaries of all earlier chapters
        if summaries:
            context_parts.append("\nSUMMARIES OF PRIOR CHAPTERS:")
            for j, summary in enumerate(summaries, 1):
                context_parts.append(f"\n  Chapter {j} summary: {summary}")

        # Add full text of immediately preceding chapter
        if i > 1:
            prev_safe_title = sanitize_filename(chapter_targets[i - 2]["title"])
            prev_filename = f"{i-1:02d} - {prev_safe_title}_generated.html"
            prev_path = gen_dir / prev_filename
            if prev_path.exists():
                prev_content = prev_path.read_text(encoding="utf-8")
                # Strip HTML wrapper, keep just body content
                prev_soup = BeautifulSoup(prev_content, "lxml")
                prev_body = prev_soup.find("body")
                prev_text = "".join(str(c) for c in prev_body.children) if prev_body else prev_content
                context_parts.append(f"\nFULL TEXT OF THE IMMEDIATELY PRECEDING CHAPTER:\n{prev_text}")

        # Chapter assignment
        is_last = (i == len(chapter_targets))
        next_chapter_note = ""
        if not is_last:
            next_ch = chapter_targets[i]
            next_chapter_note = (
                f"\nNEXT CHAPTER: \"{next_ch['title']}\" — {next_ch['description']}\n"
                f"(Use this information to write the preview paragraph at the end of your chapter.)"
            )
        else:
            next_chapter_note = (
                "\nThis is the FINAL CHAPTER of the book. Instead of a preview paragraph, "
                "end with a closing reflection on the book's overall arc and themes."
            )

        context_parts.append(
            f"\n--- YOUR ASSIGNMENT ---\n"
            f"CHAPTER NUMBER: {i} of {len(chapter_targets)}\n"
            f"CHAPTER TITLE: {ct['title']}\n"
            f"CHAPTER DESCRIPTION: {ct['description']}\n"
            f"TARGET WORD COUNT: {ct['word_target']} words (stay within 10% of this)."
            f"{next_chapter_note}"
        )

        user_message = "\n".join(context_parts)

        # Generate chapter with Opus + extended thinking
        print("  Generating…", end=" ", flush=True)
        response_text = _api_call_with_retry(
            client, gen_model, system_prompt, user_message,
            max_tokens=16384, use_extended_thinking=True, thinking_budget=10000,
        )

        if response_text is None:
            failures.append((i, ct["title"], "Max retries exceeded"))
            print("FAILED.")
            summaries.append(f"[Chapter {i} failed to generate]")
            continue

        # Process and save chapter
        response_text = markdown_to_html(response_text)
        actual_words = count_words(response_text)
        print(f"done ({actual_words:,} words).")

        # Wrap in full HTML document
        chapter_html = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="utf-8">\n'
            f"<title>{ct['title']}</title>\n"
            "</head>\n<body>\n"
            f"{response_text}\n"
            "</body>\n</html>"
        )
        chapter_path.write_text(chapter_html, encoding="utf-8")
        print(f"  Saved: {chapter_filename}")

        # Generate summary with Sonnet (cheap, no extended thinking)
        print("  Generating summary…", end=" ", flush=True)
        summary_prompt = (
            f"Write a single paragraph (~100 words) summarizing this chapter. "
            f"Focus on the key arguments and conclusions. Do not use bullet points.\n\n"
            f"{response_text}"
        )
        summary_text = _api_call_with_retry(
            client, summary_model,
            "You are a concise summarizer. Write a single paragraph summarizing the chapter provided.",
            summary_prompt,
            max_tokens=512, use_extended_thinking=False,
        )

        if summary_text:
            summary_path.write_text(summary_text.strip(), encoding="utf-8")
            summaries.append(summary_text.strip())
            print(f"done ({count_words(summary_text)} words).")
        else:
            placeholder = f"Chapter {i} covers: {ct['description'][:200]}"
            summary_path.write_text(placeholder, encoding="utf-8")
            summaries.append(placeholder)
            print("failed, using description as fallback.")

    if failures:
        print(f"\n{len(failures)} chapter(s) failed:")
        for idx, title, err in failures:
            print(f"  Ch. {idx} ({title}): {err}")
        print("Re-run with --stage 2 to retry failed chapters.")


def _api_call_with_retry(client, model, system, user_message, max_tokens=8192,
                          use_extended_thinking=False, thinking_budget=10000,
                          max_retries=5, backoff=2):
    """Make an API call with exponential backoff retry logic."""
    import anthropic

    response_text = None

    for attempt in range(max_retries):
        try:
            if use_extended_thinking:
                message = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=1,  # required for extended thinking
                    thinking={
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    },
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )
            else:
                message = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                )

            # Extract text from response (may have thinking blocks)
            for block in message.content:
                if block.type == "text":
                    response_text = block.text
                    break
            break

        except anthropic.RateLimitError:
            wait = backoff * (2 ** attempt)
            print(f"rate limited, waiting {wait}s…", end=" ", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = backoff * (2 ** attempt)
            print(f"API error ({e}), retrying in {wait}s…", end=" ", flush=True)
            time.sleep(wait)

    return response_text


# ===========================================================================
# Stage 3: Save & Validate
# ===========================================================================
def stage3(paths, chapter_targets):
    """Verify all chapter files exist and report word counts."""
    print("\n" + "=" * 60)
    print("STAGE 3: Save & Validate")
    print("=" * 60)

    gen_dir = paths["generated_chapters_dir"]
    total_words = 0
    all_present = True

    for i, ct in enumerate(chapter_targets, 1):
        safe_title = sanitize_filename(ct["title"])
        chapter_filename = f"{i:02d} - {safe_title}_generated.html"
        chapter_path = gen_dir / chapter_filename

        if chapter_path.exists():
            content = chapter_path.read_text(encoding="utf-8")
            wc = count_words(content)
            total_words += wc

            target = ct["word_target"]
            deviation = abs(wc - target) / target * 100
            flag = ""
            if deviation > 20:
                flag = " ⚠ SIGNIFICANT DEVIATION"
            elif deviation > 10:
                flag = " ⚡ slight deviation"

            print(f"  Ch. {i:2d}: {wc:,} / {target:,} words ({deviation:.0f}% off){flag}  {ct['title']}")
        else:
            print(f"  Ch. {i:2d}: MISSING  {ct['title']}")
            all_present = False

    print(f"\n  Total words: {total_words:,}")
    target_total = sum(ct["word_target"] for ct in chapter_targets)
    print(f"  Target total: {target_total:,}")

    if not all_present:
        print("\n  WARNING: Some chapters are missing. Re-run with --stage 2.")
    else:
        print(f"\n  All {len(chapter_targets)} chapters present.")


# ===========================================================================
# Stage 4: Coherence Pass
# ===========================================================================
def stage4(toc_data, style_guide, paths, chapter_targets):
    """Run coherence editing pass over generated chapters."""
    print("\n" + "=" * 60)
    print("STAGE 4: Coherence Pass")
    print("=" * 60)

    gen_dir = paths["generated_chapters_dir"]
    edit_dir = paths["coherence_edited_dir"]

    if not COHERENCE_INSTRUCTIONS_FILE.exists():
        print(f"ERROR: {COHERENCE_INSTRUCTIONS_FILE} not found.")
        sys.exit(1)
    coherence_instructions = COHERENCE_INSTRUCTIONS_FILE.read_text(encoding="utf-8").strip()

    # Read all generated chapters
    chapters = []
    for i, ct in enumerate(chapter_targets, 1):
        safe_title = sanitize_filename(ct["title"])
        chapter_filename = f"{i:02d} - {safe_title}_generated.html"
        chapter_path = gen_dir / chapter_filename

        if not chapter_path.exists():
            print(f"ERROR: Chapter {i} not found: {chapter_filename}")
            print("Run stage 2 first to generate all chapters.")
            sys.exit(1)

        content = chapter_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(content, "lxml")
        body = soup.find("body")
        body_html = "".join(str(c) for c in body.children) if body else content
        chapters.append({
            "index": i,
            "title": ct["title"],
            "safe_title": safe_title,
            "html": body_html,
        })

    num_chapters = len(chapters)
    toc_yaml = yaml.dump(toc_data, default_flow_style=False, allow_unicode=True)

    api_key = get_api_key()
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Determine windows
    if num_chapters <= 7:
        windows = [(0, num_chapters)]  # single window
        print(f"  Short book ({num_chapters} chapters) — processing in a single window.")
    else:
        windows = []
        start = 0
        while start < num_chapters:
            end = min(start + 5, num_chapters)
            windows.append((start, end))
            # Advance by 2 for overlap (windows share 3 chapters)
            if end >= num_chapters:
                break
            start += 2
        # Make sure the last window reaches the end
        if windows[-1][1] < num_chapters:
            windows.append((num_chapters - 5, num_chapters))
        print(f"  {num_chapters} chapters — processing in {len(windows)} overlapping windows.")

    print(f"  Windows: {[(w[0]+1, w[1]) for w in windows]}")

    # Track which chapters have been coherence-edited and by which window
    # For overlapping chapters, use the version from the window where they're in the middle
    edited_chapters = {}  # index -> edited html

    for win_idx, (start, end) in enumerate(windows):
        window_chapters = chapters[start:end]
        window_nums = [ch["index"] for ch in window_chapters]

        # Check cache for this window
        cache_file = edit_dir / f"window_{start+1}_{end}_cache.html"
        if cache_file.exists():
            print(f"\n  Window {win_idx+1} [{start+1}–{end}]: cached, loading…")
            cached_html = cache_file.read_text(encoding="utf-8")
            _parse_window_result(cached_html, window_chapters, edited_chapters, start, end, num_chapters)
            continue

        print(f"\n  Window {win_idx+1} [{start+1}–{end}]: generating…", flush=True)

        # Build the window's input
        chapters_text = []
        for ch in window_chapters:
            chapters_text.append(f"<h2>Chapter {ch['index']}: {ch['title']}</h2>")
            chapters_text.append(ch["html"])
            chapters_text.append("<hr/>")
        chapters_combined = "\n".join(chapters_text)

        # Identify focus chapters
        if num_chapters <= 7:
            focus_note = "Edit ALL chapters thoroughly — this is the entire book."
        else:
            middle_chapters = window_nums[1:-1] if len(window_nums) > 2 else window_nums
            edge_note = ""
            if start == 0:
                edge_note = f" Chapter {window_nums[0]} is the first chapter of the book — edit it fully."
                middle_chapters = window_nums[:len(window_nums)-1]
            if end == num_chapters:
                edge_note += f" Chapter {window_nums[-1]} is the final chapter — edit it fully."
                middle_chapters = window_nums[1:]
            focus_note = (
                f"Focus your edits primarily on chapters {middle_chapters} "
                f"(the middle of this window). Lighter touch on edge chapters "
                f"as they'll be covered by adjacent windows.{edge_note}"
            )

        system_prompt = coherence_instructions + "\n\n--- STYLE GUIDE ---\n\n" + style_guide

        user_message = (
            f"BOOK TITLE: {toc_data['title']}\n"
            f"AUTHOR: {toc_data.get('author', 'Unknown')}\n"
            f"TOTAL CHAPTERS IN BOOK: {num_chapters}\n"
            f"THIS WINDOW: Chapters {window_nums[0]}–{window_nums[-1]}\n\n"
            f"FOCUS: {focus_note}\n\n"
            f"TABLE OF CONTENTS:\n{toc_yaml}\n\n"
            f"CHAPTERS TO EDIT:\n\n{chapters_combined}"
        )

        # Stream the response (these are large)
        result_text = _stream_api_call_with_retry(
            client, "claude-opus-4-6", system_prompt, user_message,
            max_tokens=32768, thinking_budget=10000,
        )

        if result_text is None:
            print(f"  Window {win_idx+1} FAILED. Using original chapters.")
            for ch in window_chapters:
                if ch["index"] not in edited_chapters:
                    edited_chapters[ch["index"]] = ch["html"]
            continue

        result_text = markdown_to_html(result_text)
        cache_file.write_text(result_text, encoding="utf-8")
        _parse_window_result(result_text, window_chapters, edited_chapters, start, end, num_chapters)

    # Save all coherence-edited chapters
    print("\n  Saving coherence-edited chapters…")
    for i, ct in enumerate(chapter_targets, 1):
        safe_title = sanitize_filename(ct["title"])
        edited_filename = f"{i:02d} - {safe_title}_edited.html"
        edited_path = edit_dir / edited_filename

        html_content = edited_chapters.get(i)
        if html_content is None:
            # Fallback to original
            orig_filename = f"{i:02d} - {safe_title}_generated.html"
            orig_path = gen_dir / orig_filename
            html_content = orig_path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html_content, "lxml")
            body = soup.find("body")
            html_content = "".join(str(c) for c in body.children) if body else html_content

        full_html = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="utf-8">\n'
            f"<title>{ct['title']}</title>\n"
            "</head>\n<body>\n"
            f"{html_content}\n"
            "</body>\n</html>"
        )
        edited_path.write_text(full_html, encoding="utf-8")
        wc = count_words(html_content)
        print(f"    {edited_filename} ({wc:,} words)")


def _parse_window_result(html, window_chapters, edited_chapters, start, end, num_chapters):
    """Parse the coherence pass result and assign chapters to the edited_chapters dict."""
    soup = BeautifulSoup(html, "lxml")
    h2_tags = soup.find_all("h2")

    if not h2_tags:
        # No h2 tags found — treat as a single block and assign to all window chapters
        for ch in window_chapters:
            if ch["index"] not in edited_chapters:
                edited_chapters[ch["index"]] = html
        return

    # Extract each chapter's content between h2 tags
    parsed_chapters = []
    for j, h2 in enumerate(h2_tags):
        parts = [str(h2)]
        for sibling in h2.find_next_siblings():
            if sibling.name == "h2":
                break
            if sibling.name == "hr":
                continue
            parts.append(str(sibling))
        parsed_chapters.append("\n".join(parts))

    # Map parsed chapters to window chapters
    for j, ch in enumerate(window_chapters):
        if j < len(parsed_chapters):
            # For overlapping windows, prefer the version where this chapter is in the middle
            is_middle = True
            if num_chapters > 7:
                if start > 0 and j == 0:
                    is_middle = False  # first chapter in non-first window — edge
                if end < num_chapters and j == len(window_chapters) - 1:
                    is_middle = False  # last chapter in non-last window — edge

            if is_middle or ch["index"] not in edited_chapters:
                edited_chapters[ch["index"]] = parsed_chapters[j]


def _stream_api_call_with_retry(client, model, system, user_message,
                                 max_tokens=32768, thinking_budget=10000,
                                 max_retries=5, backoff=2):
    """Stream an API call with extended thinking and retry logic."""
    import anthropic

    for attempt in range(max_retries):
        try:
            collected_text = []
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=1,  # required for extended thinking
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                },
                system=system,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == 'content_block_delta':
                            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                                collected_text.append(event.delta.text)
                                if sum(len(t) for t in collected_text) % 2000 < len(event.delta.text):
                                    print(".", end="", flush=True)
            print()
            return "".join(collected_text)

        except anthropic.RateLimitError:
            wait = backoff * (2 ** attempt)
            print(f"rate limited, waiting {wait}s…", end=" ", flush=True)
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = backoff * (2 ** attempt)
            print(f"API error ({e}), retrying in {wait}s…", end=" ", flush=True)
            time.sleep(wait)

    return None


# ===========================================================================
# Stage 5: Assemble EPUB + HTML
# ===========================================================================
def stage5(toc_data, paths, chapter_targets):
    """Assemble the final EPUB and HTML from coherence-edited chapters."""
    print("\n" + "=" * 60)
    print("STAGE 5: Assemble EPUB + HTML")
    print("=" * 60)

    from ebooklib import epub

    edit_dir = paths["coherence_edited_dir"]
    output_dir = paths["output_dir"]

    book_title = toc_data["title"]
    book_author = toc_data.get("author", "Unknown")
    safe_title = re.sub(r"\s+", "_", re.sub(r"[^\w\s\-]", "", book_title).strip())

    # Read all edited chapters
    chapter_contents = []
    total_words = 0

    for i, ct in enumerate(chapter_targets, 1):
        safe_ch_title = sanitize_filename(ct["title"])
        edited_filename = f"{i:02d} - {safe_ch_title}_edited.html"
        edited_path = edit_dir / edited_filename

        if not edited_path.exists():
            # Fall back to generated chapter
            gen_filename = f"{i:02d} - {safe_ch_title}_generated.html"
            gen_path = paths["generated_chapters_dir"] / gen_filename
            if gen_path.exists():
                print(f"  WARNING: Using unedited chapter {i} (coherence-edited version missing).")
                edited_path = gen_path
            else:
                print(f"  ERROR: Chapter {i} not found in either directory.")
                sys.exit(1)

        content = edited_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(content, "lxml")
        body = soup.find("body")
        body_html = "".join(str(c) for c in body.children) if body else content
        wc = count_words(body_html)
        total_words += wc
        chapter_contents.append({
            "index": i,
            "title": ct["title"],
            "html": body_html,
            "word_count": wc,
        })
        print(f"  Ch. {i:2d}: {wc:,} words — {ct['title']}")

    print(f"\n  Total: {total_words:,} words")

    # Build EPUB
    book = epub.EpubBook()
    book.set_identifier("ebook-generated-" + re.sub(r"\W+", "-", book_title.lower()))
    book.set_title(book_title)
    book.set_language("en")
    book.add_author(book_author)

    css_content = """
body { font-family: Georgia, "Times New Roman", serif; line-height: 1.6; margin: 1em; color: #222; }
h1, h2, h3, h4 { margin-top: 1.5em; margin-bottom: 0.5em; line-height: 1.2; }
h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
h3 { color: #444; }
p { margin-bottom: 0.8em; text-align: justify; }
blockquote { margin: 1em 2em; font-style: italic; color: #555; border-left: 3px solid #ccc; padding-left: 1em; }
hr { border: none; border-top: 1px solid #ccc; margin: 2em 0; }
"""
    css = epub.EpubItem(
        uid="style", file_name="style/default.css",
        media_type="text/css", content=css_content.encode("utf-8"),
    )
    book.add_item(css)

    epub_chapters = []
    toc = []

    for ch in chapter_contents:
        chap = epub.EpubHtml(
            title=ch["title"],
            file_name=f"chapter_{ch['index']:02d}.xhtml",
            lang="en",
        )
        chap.content = (
            f'<html><head><link rel="stylesheet" href="style/default.css" '
            f'type="text/css"/></head><body>{ch["html"]}</body></html>'
        ).encode("utf-8")
        chap.add_item(css)
        book.add_item(chap)
        epub_chapters.append(chap)
        toc.append(epub.Link(
            f"chapter_{ch['index']:02d}.xhtml",
            ch["title"],
            f"ch{ch['index']}",
        ))

    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    epub_path = output_dir / f"{safe_title}.epub"
    epub.write_epub(str(epub_path), book)
    print(f"\n  EPUB saved: {epub_path}")

    # Build combined HTML
    combined_parts = [f"<h1>{book_title}</h1>"]
    for ch in chapter_contents:
        combined_parts.append(ch["html"])
        combined_parts.append("<hr/>")
    combined_html = "\n".join(combined_parts)

    html_doc = (
        f"<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        f"<title>{book_title}</title>\n<style>{css_content}</style>\n"
        f"</head>\n<body>\n{combined_html}\n</body>\n</html>"
    )
    html_path = output_dir / f"{safe_title}.html"
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"  HTML saved: {html_path}")
    print(f"\n  Final word count: {total_words:,}")

    return epub_path


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Book Generator — Write a full book from a TOC and style guide",
    )
    parser.add_argument("toc", help="Path to table of contents YAML file")
    parser.add_argument("style_guide", help="Path to style guide text file")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help="Start from this stage (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse inputs and show plan, but don't call API")
    parser.add_argument("--target", type=int, default=None,
                        help=f"Override word target per chapter (default: {DEFAULT_WORD_TARGET})")
    parser.add_argument("--regen", type=int, default=None, metavar="N",
                        help="Regenerate chapter N (delete cached files and re-run stage 2)")

    args = parser.parse_args()

    toc_path = Path(args.toc).resolve()
    style_path = Path(args.style_guide).resolve()

    print("=" * 60)
    print("  Book Generator")
    print("=" * 60)

    if args.dry_run:
        print("  Mode: DRY RUN")
    print(f"  Starting from stage: {args.stage}")

    # Stage 1 always runs (to load inputs)
    toc_data, style_guide, paths, chapter_targets = stage1(
        toc_path, style_path, args.target,
    )

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE — No API calls made.")
        print("=" * 60)
        return

    # Confirm before proceeding
    if args.stage <= 2:
        print()
        confirm = input("Proceed with API calls? [Y/n] ").strip()
        if confirm.lower() == "n":
            print("Aborted by user.")
            sys.exit(0)

    if args.stage <= 2:
        stage2(toc_data, style_guide, paths, chapter_targets, regen_chapter=args.regen)

    if args.stage <= 3:
        stage3(paths, chapter_targets)

    if args.stage <= 4:
        stage4(toc_data, style_guide, paths, chapter_targets)

    if args.stage <= 5:
        stage5(toc_data, paths, chapter_targets)

    print("\n" + "=" * 60)
    print("  Book generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
