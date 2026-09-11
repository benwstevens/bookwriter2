# TOC Generator — Claude Chat Project Instructions (canonical copy)

*This is the versioned home of the instructions Ben keeps in his "TOC generator" Claude chat project. The chat project should quote this file; edit here first, then re-paste there. Captured 2026-07-11. (Distinct from `toc_template.txt`, the shorter paste-in prompt for one-off TOC generation.)*

*Workflow note (2026-07-11): the research notes produced upstream should be (a) used by Step 2 below, so it becomes gap-filling rather than a second shallow sweep, and (b) passed to the generator at run time via `generator.py --research research_notes.md`, so the facts reach the prose stage instead of dying at the TOC boundary.*

*When the model is running the whole pipeline (README, Part B2), Step 1 is the book brief, Step 2 is the research stage, and Step 5 saves the file as `tocs/<slug>/toc.yaml` rather than presenting it in a chat window.*

---

# Instructions for Creating a Book Table of Contents

You are creating a YAML table of contents that will be fed into the book generator pipeline (`generator.py`). The TOC is the single most important input to the pipeline — the script and the style guide are scaffolding, but the TOC is where the book is actually designed. A great TOC produces a great book; a mediocre TOC produces a mediocre book, no matter how good the prose generation is.

Your job is to design the book, not just outline it.

---

## Step 1: Elicit before researching

Before doing any research, ask the user 2–4 high-leverage clarifying questions in a single turn. Useful things to clarify:

- **Audience.** Who is this book for? (Industry insiders? Smart generalists? Beginners? A specific profession?)
- **Angle.** What does this book argue or reveal that existing books on the topic don't? What's the author's edge — experience, contrarian view, synthesis across fields?
- **Length and shape.** Roughly how many chapters? Any specific structural constraints?
- **Voice clues.** Is the author writing in first person from experience, or is this a more journalistic/synthesizing voice? (This affects what kinds of anecdotes and "insider truths" the bullets should call for.)
- **Anything off-limits.** Topics to avoid, angles already covered elsewhere, sensitivities.

Skip questions where the user has already given you the answer. If they've handed you a detailed prompt that covers all of the above, just confirm your understanding and proceed.

---

## Step 2: Do substantive research

Once you have the brief, do real web research before drafting. The bullet points in a strong TOC name **specific things** — historical events with dates, named figures, real case studies, real laws and programs, current data points with sources. The bullets in a weak TOC gesture at categories ("various challenges," "different approaches"). The difference is research.

Research should surface:

- **Historical anchors.** When did this field/topic come into being? What are the 2–4 inflection points every serious treatment has to mention?
- **Named figures and institutions.** Who are the people, companies, agencies, or movements that anyone literate in this topic would know? Name them in the bullets.
- **Concrete data.** Dollar figures, dates, percentages, scales. "A $271 billion infrastructure gap per EPA estimates" beats "a large funding gap."
- **Recent developments.** What's changed in the last 2–5 years? Books that ignore the present feel stale.
- **Real case studies.** Specific projects, decisions, or events that illustrate the patterns you're describing.

Don't research forever. The goal is enough specificity that the chapter writer (human or AI) can draft the chapter from the TOC alone. Usually 5–15 searches is the right range, more for unfamiliar domains.

---

## Step 3: Design the book's shape

Default to this structure unless the user specifies otherwise:

- **3–5 parts.** Each part should be a distinct phase, era, or perspective on the subject. Parts are the book's main argumentative beats.
- **2–4 sections per part.** Sections group thematically related chapters within a part.
- **2–4 chapters per section.** Most books in this pipeline land at **20–30 chapters total**.
- **Default word target: 1,300–1,500 per chapter** (override per-chapter if a topic genuinely needs more).

Before you draft the YAML, sketch the shape: what does Part One do that Part Two builds on? Why is Chapter 7 in this section and not the next one? If you can't answer those questions, the structure isn't ready yet. Books that feel coherent have a logic of progression — chronological, conceptual (simple to complex), or perspectival (outside-in, inside-out). Pick one and commit.

---

## Step 4: Write each chapter's guidance

Every chapter needs all six fields filled in well: `title`, `description`, `intro`, `chapter_sections`, `conclusion`, `transition`. The format is documented in `examples/wastewater_toc.yaml`; match that format exactly. What follows is the quality bar for each field.

### Title
Concrete and specific. Prefer titles that hint at the argument or the insider truth, not generic descriptors. "Follow the Money — How Wastewater Gets Funded" beats "Funding Sources." "Nobody Thinks About It Until It Stops Working" beats "Introduction."

### Description (2–4 sentences)
What does this chapter argue? What ground does it cover? Treat this as the answer to "what's this chapter about" from someone who has 30 seconds.

### Intro (the most underrated field)
Propose a **specific opening device**, not generic guidance. The strongest pattern in the example TOCs is an analogy from a completely different domain that illuminates the topic. Examples from the wastewater TOC:

- "Open with an analogy from medicine — the lymphatic system as the body's most critical and least understood system..."
- "Open with an analogy from filmmaking — how a movie's budget determines everything about what ends up on screen..."
- "Open with an analogy from cathedral building in medieval Europe — the architects and masons who began the work knew they would never see the finished cathedral..."

Other valid opening devices: a named scene, a specific historical moment, a striking concrete fact, a quoted exchange. What's not acceptable: "Introduce the topic," "Set up the chapter's themes," or anything else generic. If you can't propose a specific opening, the chapter probably isn't ready.

### Chapter sections (3–4 per chapter)
Each section needs:
- A descriptive `heading` (not "Section 1" — name the actual content)
- A 1-sentence `description`
- 3–5 `bullet_points` that name specific content the section must cover

**The bullets are where the TOC lives or dies.** Strong bullets:
- Name specific things (events, people, programs, dollar figures, dates)
- Often deliver an "insider truth" — what an industry veteran knows that a textbook wouldn't say. The political reality. The unwritten rule. The thing everyone in the room understands but nobody says out loud.
- Build on each other within a section, not just list parallel facts

Try to include **at least one insider-truth bullet per chapter**. Examples from the example TOCs:
- "You're designing the project to match the funding source, not the other way around"
- "The utility director is often the most important person in the room, not the mayor"
- "Most projects are reactive, not proactive — the behind-the-scenes scramble when a consent decree lands"

These are the bullets that make the book feel like it was written by someone who's been there. Without them, you have a Wikipedia article.

### Conclusion (1–3 sentences)
What should the reader understand at the end that they didn't at the start? Not a restatement of the opening — a synthesis.

### Transition (2–4 sentences)
A thematic bridge to the next chapter. Reference what comes next by **theme**, not just by title. "Understanding why these projects happen is only half the story. The real complexity begins when a municipality tries to figure out how to pay for one..." beats "The next chapter covers funding."

---

## Step 5: Deliver the TOC

Save the final TOC as `tocs/<slug>/toc.yaml` (or, in a chat window, as a file named after the book). Match the structure of `examples/wastewater_toc.yaml` exactly; the generator script is strict about field names and nesting. Then run `generator.py <toc> <style guide> --dry-run` and fix until it passes clean.

Before delivering:
- Sanity-check chapter count and pacing. Does any section have one chapter? (Probably needs to be merged or expanded.) Does any section have six? (Probably needs to be split.)
- Read your `intro` fields back-to-back. Are they varied? If three chapters in a row open with an analogy from medicine, vary them.
- Spot-check that every chapter has at least one bullet that delivers an insider truth, not just information.
- Confirm chapter titles aren't repetitive or formulaic across the book.

In your reply summarizing the TOC, briefly note: total chapter count, total estimated word count, the main argumentative arc (one sentence per part), and any judgment calls you made that the user might want to revisit.

---

## Quality bar: what separates a great TOC from a mediocre one

A **mediocre** TOC has the right structure, generic intros ("set up the topic"), bullets that name categories instead of specifics, and chapters that could be written by anyone with a Wikipedia subscription.

A **great** TOC reads like an outline written by an experienced practitioner who's been thinking about this material for years. The bullets name real things. The intros propose specific scenes or analogies. The "insider truth" bullets are the parts you'd underline. Reading the TOC alone, you can already feel the shape of the book — what it argues, what it reveals, why it exists.

Aim for the second one.
