# bookwriter2

A tool that writes a whole non-fiction book from a chapter-by-chapter plan. Give your AI a topic, a reader, and an angle; it designs the plan, you approve it, and the tool writes every chapter in one voice and hands you an EPUB.

## For the person

*This section is for you. Everything from "For the model" onward is instructions for your AI tool, telling it how to set this up and how to write a book with you. The last section, "For maintainers," is about the code.*

**Before anything else: what you need.** Four things. First, an AI tool that runs on your computer and can run programs and read files: Claude Code today, or something similar. If you use ChatGPT or Claude in a web browser, this is not for you yet; a browser chat cannot run the script. Second, an Anthropic API key with billing turned on. This is separate from a Claude subscription: the tool bills your API account for each book, and a book has cost between about four and thirty-five dollars on the runs so far. You always see an estimate before anything is spent, and nothing is spent until you say go. Third, Python 3.10 or newer; the model checks for it and installs what is missing, with your permission. Fourth, something that opens an EPUB: a Kindle, Apple Books, or just a web browser for the HTML copy it also makes. To remove it: delete the folder. The books it made are ordinary files and stay yours.

**Why bother.** The book you want to read has not been written. It is too niche, too new, or pitched at the wrong reader. Give this a topic, a reader, and an angle. It designs a chapter-by-chapter plan, you read the plan and approve it, and it writes the whole book in one voice and hands you an EPUB the same afternoon. People have used it for a developer's guide to building a wastewater plant, a kids' guide to how doctors are made, a father-and-son priority list for learning golf built only from what has been measured, a field guide to one city's rent laws, and a forty-chapter book that walks a twelve-year-old through building a small language model.

**The idea in five lines.**
1. The plan is the book. The table of contents carries every argument, every example, and every opening device; the writing stage is mechanical.
2. The model designs the plan. You never write a line of it. You answer a handful of questions and read a chapter list.
3. Nothing is spent until you say go, and the price comes first.
4. It writes in a voice built for you from a short questionnaire, on top of a fixed set of readability rules (short chapters, short sentences, short paragraphs, a clear opener, and an "in the next chapter" teaser) that are the reason people finish these books.
5. For anything recent or niche, it researches first, and the book is only as true as that research.

**Two honest things.** First: it writes convincing prose, not checked facts. There is no fact-checking stage. Read the result as a very well-organised first draft by a smart stranger, and do not act on a number in it without checking. The model will offer a manual check at the end, and you should take it for any book you will make decisions from. Second: it is for explainer non-fiction. It is not for fiction, not for memoir, and not for a book that needs your own stories unless you hand them over when it asks.

**What happens next.** Open your AI tool in this folder and say "set up the book generator." That is a one-time conversation of about fifteen minutes. It installs what is needed, walks you through getting an API key (you paste the key into a file yourself; the model never handles it), picks a default voice with you, and runs a three-chapter test book for well under a dollar so you see the whole thing work. After that, say "write me a book about ..." whenever you like. It asks four or five questions, researches if the topic needs it, shows you a chapter list and a price, and writes the book after you say go. Budget an hour of your own attention and one to three hours of the machine's. (In Claude Code, `/setup-bookwriter` and `/write-a-book` do the same two things.)

## Where this comes from

bookwriter2 was built by **Ben Stevens** (benwstevens.com) in February 2026 as the reverse of a book distiller he already had: instead of condensing a book that exists, write one that does not. He has generated more than twenty books with it since, for work, for his kids, and for himself, and four lessons from those runs shape this file. The table of contents is eighty percent of the quality, so the model spends its effort there. Research has to reach the writing stage, or the facts die at the outline. One "insider truth" per chapter, the thing a veteran knows and a textbook does not say, is what separates a book from an encyclopedia entry. And the model runs long, so targets are set a little low. Underneath every voice sits one skeleton he arrived at by reading his own output: chapters under 1,500 words, no sentence over 25 words, no paragraph over 200, three to five sections of four to five paragraphs, a two-to-three paragraph opener, a conclusion, and a teaser for the next chapter. That skeleton is the readability secret and ships as the default; the questionnaire builds the voice around it. Some of what shipped with it is one person's house style (no em-dashes; a file of book-wide constants for long books); the setup interview offers those as options and does not impose them.

---

## For the model

**If you are the model:** everything from here to the vocabulary is addressed to you. Read this whole file, then `toc_project_instructions.md` (how to design a table of contents that produces a good book), then `generator_instructions.txt` and `generator_coherence_instructions.txt` (what the script tells the API). There are two interviews. The **setup interview** (Part B1) happens once and writes `config.yaml`. The **book brief** (Part B2) happens for every book and writes `tocs/<slug>/brief.md`. Write answers to the file as they land, so a dropped session loses nothing. Do not skip a question because you can guess the answer; the point is that the person hears the choices. One exception to any standing rule about ending a message with a recap of open requests: skip it during an interview round. The round *is* the questions.

## Part A. What the generator is

**In plain words.** `generator.py` takes two files, a table of contents in YAML and a style guide in plain text, and writes the book in five stages. It checks the plan and prices the book. It writes the chapters in order, each one seeing the whole outline, a summary of every earlier chapter, and the full text of the chapter just before it. It checks the word counts. It re-reads neighbouring chapters together and smooths the seams. It packages an EPUB and an HTML file. Every stage saves its work, so a run that fails picks up where it stopped, and one chapter can be rewritten on its own. If research notes are given, they ride along with every chapter as the book's factual ground truth.

**Say these things unprompted, in these words or close to them.**
- "Nothing is spent until you say go. Before every book I run a rehearsal that checks the plan and shows you the estimate, and I never start writing without a clear yes from you in this conversation."
- "I design the plan and you approve it. You never touch the YAML; you read a chapter list."
- "There is no fact-check stage. When the book is done I will offer to go through it against the research and flag every number, name, and date I cannot support. Until then, treat every fact in it as unchecked."
- "The estimate runs high and the chapters run long. On the biggest book so far the estimate was more than double the bill, and chapters came in ten to forty percent over their targets."

**The words, defined once here and used freely after.** A **TOC** (table of contents) is the chapter-by-chapter plan; **YAML** is the plain-text format it is written in, which the person reads and you write. A **style guide** is a one-page description of the voice. **Research notes** are facts gathered before the plan so the book does not invent them. A **dry run** is the rehearsal that checks the plan and prices the book without spending. A **stage** is one of the five steps; a failed run restarts at the stage that failed. The **coherence pass** is the second read of neighbouring chapters. **Regen** rewrites one chapter. A **slug** is the short folder name made from the title. An **EPUB** is the e-book file. An **API key** is the password-like string that lets the script bill the person's account. A **token** is the unit the bill is counted in, about three-quarters of a word. A **virtual environment** is a private copy of Python so this tool cannot disturb anything else on the machine.

---

## Part B1. The setup interview (once)

Eight rounds, two or three questions each, in plain language, with a default on every question so the person can say "default." Write each answer into `config.yaml` (copy `config.example.yaml` first) as it lands. Do not install anything, and do not spend anything, without asking in stated words first.

### 1. Tool and folder
- You can usually see which tool you are running in; confirm it rather than asking: "I'm running in Claude Code, so I can run the script and read the files here."
- "Is this folder inside something that syncs, like Dropbox, iCloud, OneDrive, or Google Drive?" If yes, the Python environment goes outside it (round 2), because a synced environment fetches thousands of files one at a time and can hang for minutes.
- "Where should finished books land?" *(Default: a `books/` folder beside the script.)* Scripted: "I will not move or rename anything you already have."

### 2. Python and the environment
Check the Python version and say what you found. Then ask permission in these words or close to them: **"I'd like to create a private Python environment for this tool and install six packages into it. That touches nothing else on your machine. If your folder is synced, I'll put the environment outside it. May I?"** *(Default: yes, at `~/.venvs/bookwriter/`.)* On yes: create it, `pip install -r requirements.txt`, and record the path under `venv` in `config.yaml`.

### 3. The API key
Explain in two sentences: "An API key is a long password that lets a program bill your Anthropic account directly. It is separate from any Claude subscription, and it is what pays for the books." Send the person to console.anthropic.com, Settings, API keys, with billing enabled, and ask them to paste the key into the `.env` file themselves as `ANTHROPIC_API_KEY=...`. Do not take the key in chat, do not print it, do not read `.env` back aloud. Confirm `.env` is git-ignored (it is). State the cost range plainly: "Books so far have cost between about four and thirty-five dollars each, and you will see an estimate before every one."

### 4. The spending rule
"Before any book, I run a dry run and show you the estimate. I never start writing without a yes from you in that conversation. Do you want a ceiling above which I ask a second time?" *(Default: twenty dollars.)* Record it as `spend_ceiling`.

### 5. Reading device
"Where will you read these: a Kindle, Apple Books, or a browser?" *(Default: browser.)* Record it as `reading_device` and say once how the file will get there: Send to Kindle by email or drag-and-drop for a Kindle; double-click for Apple Books; open the HTML file for a browser.

### 6. The voice: a short questionnaire
This is the round that decides whether the books read well, so it gets six small steps instead of one question. Open by saying, in these words or close to them: **"Every book this writes sits on one fixed skeleton, and the skeleton is the reason people finish them: chapters of about 1,300 words and never more than 1,500; no sentence over 25 words; no paragraph over 200; three to five sections of four to five paragraphs; an opener of two or three paragraphs; a short conclusion; and an 'in the next chapter' teaser. I'll keep that unless you tell me to change it. What I need from you is the voice on top of it."** Then ask the steps below two or three questions at a time, with a default on each. Write the answers into `style_guides/default.txt`, built from `style_guides/TEMPLATE.txt`, as they land. The four shipped guides in `style_guides/` are finished examples; show the one nearest the person's answers when it helps.

- **6a. Who is speaking, who is reading.** "Who is the author: a practitioner with years in the field, a curious outsider who did the reading, or a guide building the thing alongside the reader? First person or third?" *(Default: a practitioner, first person.)* "Who reads it, what do they already know, and where do they read: phone, bed, desk?" *(Default: a smart generalist outside the field, on a phone.)* If there is a second reader (a kid and a grown-up; a client and an advisor), record how the text speaks to each.
- **6b. Register and reading level.** "Which is closest: a mentor over coffee; a smart friend at dinner; a mischievous guide building it with you; a plain explainer?" *(Default: mentor over coffee.)* "What grade level?" *(Default: tenth; eighth for a hurried phone reader; seventh to eighth for a kid.)*
- **6c. Stories, metaphors, numbers.** "Real stories only, or composites if flagged? Whose stories?" *(Default: real, composites flagged, the author's own.)* "Where should metaphors come from, and should every metaphor say where it breaks down?" *(Default: everyday life and fields far from the subject; yes, say where it breaks.)* "Should big numbers be anchored to something physical?" *(Default: yes.)*
- **6d. Evidence and honesty.** "Should claims be labeled by how well they are established: well established, contested, plausible but unmeasured?" *(Default: no labels for a practitioner's book; yes for an evidence-first one.)* "Sources named in the sentence?" *(Default: when a figure matters.)* "How should the book handle what nobody knows?" *(Default: say so plainly.)*
- **6e. Humor, bans, recurring pieces.** "What kind of humor, and what is it never at the expense of?" *(Default: light and wry; never a group of people.)* "Anything I should never do? People often say: no em-dashes; no hype words like 'leverage' or 'journey'; no equations; nothing that needs a calculator." *(Default: the filler and cliché bans only.)* "Any fixed piece every chapter should carry, such as a sidebar for a second reader, a 'try this' task, or a myth box?" *(Default: none.)*
- **6f. The skeleton, confirmed.** Read the readability rules back and ask whether to loosen or tighten any: a technical reference might want longer chapters, a kid's book shorter ones. *(Default: as shipped.)* Record any change in the READABILITY block and in `config.yaml` under `house_rules`, so it survives a re-copy.

Then assemble the guide, show it in full, take corrections, and save it. Bans also go into `config.yaml` under `house_rules`.

### 7. Research
"Do I have a way to search the web from here, or a deep-research capability?" If yes, say that books on recent or niche topics will get a research pass first, at no API cost to the script, and that paywalled sources are mostly out of reach. If no, say what that means: "Books on well-known, older topics will be fine. For anything recent or niche I will tell you the book needs research notes, and you can paste them in or we can skip the topic."

### 8. The test run
Say: "Last step. I'll run the shipped three-chapter test book so you see the whole thing work. First the dry run." Run `generator.py examples/test_toc.yaml examples/test_style_guide.txt --dry-run`, show the estimate (well under a dollar), and ask for the go. On yes, run it in the foreground with `--yes`, narrate the stages as they print, then open the output folder and confirm the person can see the book on their chosen device. Ask whether to keep or delete the test book.

Then read `config.yaml` back in full, take corrections, and say what was built (Part C).

---

## Part B2. The book brief (every book)

One turn of questions, skipping anything the person's idea already answers; then the model works alone; then one approval message; then generation. The questions come from `toc_project_instructions.md` Step 1, sharpened by use.

### The questions, in one turn
1. **The question the book answers.** "In one sentence, what will a reader be able to do or understand at the end that they cannot now?" Prime with two examples: "explain a bakery from the oven door"; "walk into a rent-stabilised building and know which exits are real."
2. **The reader.** Who, what they already know, and whether there is a reader pair (a kid and a parent; a client and their advisor). *(Default: a smart generalist outside the field.)*
3. **The angle and the edge.** What it argues that existing books do not, and whether the author writes from experience (first person, war stories) or synthesises. *(Default: synthesising, third person, unless the default style guide says otherwise.)*
4. **Length and shape.** Chapters, and words per chapter. *(Default: twenty to thirty chapters at 1,300 words; say the cost band that implies, roughly ten to twenty dollars.)* For a how-to book, ask whether chapters should alternate doing and explaining.
5. **Off-limits and must-includes.** Sensitivities, topics to skip, and anything the person insists appears. Prime: "people often say: no jargon without a definition; nothing the reader would need a calculator for; the last chapter has to be practical."
6. **Research.** Decide and say: "This topic is recent or specialised enough that I want to research it first; that adds about twenty minutes and costs nothing on the script's bill" or "This is well-trodden and I will write the plan from what I know." The person can override either way. If research is on, say where it will be saved and that paywalled sources are mostly unreachable.
7. **Voice.** The default guide, or adapt it for this book. If adapting, run the short form of the questionnaire (who is speaking, who is reading, register, bans; the skeleton stays), draft the adapted guide, and show the changed lines.

Write the answers to `tocs/<slug>/brief.md`.

### The work, without further questions
- **Research**, if agreed: fan out across the topic's history, named figures and institutions, concrete data with sources, the last two to five years, and real cases. Save to `tocs/<slug>/research_notes.md` with a source list and verification flags. Follow the research process in `toc_project_instructions.md` Step 2.
- **Design the outline** per `toc_project_instructions.md` Steps 3 and 4. Bullets carry the facts verbatim from the research. Every chapter has a specific opening device and at least one insider-truth bullet. Save to `tocs/<slug>/toc.yaml` and copy the style guide to `tocs/<slug>/style_guide.txt`.
- **For long or exact books** (above about twenty-five chapters, or anything that teaches something with real numbers, commands, or a named recurring thing), add a design step before the YAML: a chapter list with a block of **book-wide constants** every chapter must obey (names, sizes, figures, banned words), and afterwards a continuity pass that reads the YAML against those constants and fixes dependency order, contradicting numbers, and transitions that point at the wrong chapter. Keep a backup of the YAML before the pass.
- **Loop on the dry run.** Run `generator.py tocs/<slug>/toc.yaml tocs/<slug>/style_guide.txt --dry-run` (add `--research tocs/<slug>/research_notes.md` if notes exist) and fix the YAML until validation passes clean. The dry run also prices the book.

### The go
One message the person can read on a phone: the chapter count and total words; one sentence of arc per part; the judgment calls worth revisiting; the estimate, with the reminder that estimates run high; and the title, offered with two alternatives, because the title is the person's call. Then: "Shall I write it?" If the estimate is above `spend_ceiling`, ask a second time in a separate message. Nothing runs before the yes.

### Generation and delivery
- Run in the background with the environment from `config.yaml` and `--yes` (the yes was given in chat; the flag only skips the terminal prompt). Twenty to thirty chapters take one to two hours. Report progress as chapters land.
- On failure or interruption, resume with `--stage 2`; finished chapters are skipped. A chapter the person dislikes is rewritten with `--regen N`, then `--stage 4` re-runs the coherence pass.
- When done: check the word-count report, confirm every chapter is present, spot-read three chapters for coherence-pass damage, and hand over the path and how to open it on the chosen device.
- Offer the fact check: "Shall I go through it against the research and flag every number, name, and date I cannot support?" For any book the person will act on, recommend it.
- Write a short session log in `tocs/<slug>/Logs/`: what was asked, what was produced, the cost, what is open.

---

## Part C. What to build and what to say

**Setup builds:** the environment at the recorded path; `.env` with the person's key (they paste it); `config.yaml` from `config.example.yaml` with every answer filled in; `style_guides/default.txt` built from the questionnaire on `style_guides/TEMPLATE.txt`; an empty `tocs/` folder; and the test book's output. Nothing else on the machine is touched.

**Each book builds:** `tocs/<slug>/` holding `brief.md`, `research_notes.md` if any, `toc.yaml`, `style_guide.txt`, `generation.log`, and `Logs/`; and under the output folder, `<slug>/` holding the generated chapters, the coherence-edited chapters, and `output/` with the EPUB and HTML.

**First-run checklist, done with the person watching:** the dry run prints a chapter list and an estimate; the go is asked for; chapters appear one at a time in the output folder; the EPUB opens on the chosen device; a log lands. Then keep or delete the test book.

**What to say at the end of setup:** where `config.yaml` is and that it is theirs to edit; the one phrase that starts a book; that the estimate always comes first and nothing runs without their yes; that nothing is checked for truth until they ask; and that the style guide is a plain text file they can open and change any time.

---

## Vocabulary, for the person reading over the model's shoulder

**TOC:** the chapter-by-chapter plan. **YAML:** the plain-text format the plan is written in; you read it, the model writes it. **Style guide:** a one-page description of the voice. **Research notes:** facts gathered before the plan, so the book does not invent them. **Dry run:** a rehearsal that checks the plan and prices the book without spending. **Stage:** one of five steps; a failed run restarts at the stage that failed. **Coherence pass:** a second read of neighbouring chapters to smooth seams. **Regen:** rewrite one chapter. **Slug:** the short folder name made from the title. **EPUB:** the e-book file. **API key:** the password-like string that lets the tool bill your account. **Token:** the unit the bill is counted in, about three-quarters of a word. **Virtual environment:** a private copy of Python so this tool cannot disturb anything else.

## A note on EPUB files, for whoever wants it

An EPUB is the standard e-book file, the same kind you buy from any bookstore except Amazon's. To read one on a **Kindle**, email it to your Send to Kindle address or drag it onto the Send to Kindle page in a browser; it shows up on the device in a minute or two. On a **Mac, iPhone, or iPad**, double-click it or tap it and Apple Books opens it. On **Android**, Google Play Books or any reader app opens it. In any **browser**, open the HTML file that lands beside the EPUB; it is the same book as one long web page. **Calibre**, a free program, converts an EPUB to anything else. None of this is programming.

---

## For maintainers

### Repo map
| Path | What it is |
|---|---|
| `generator.py` | The five-stage pipeline. Model pins are the three constants at the top. |
| `shared.py` | Utilities shared with the sibling book-distiller project (API key, word counts, EPUB helpers). |
| `generator_instructions.txt` | The system prompt for chapter writing. Carries the readability skeleton (1,500-word cap, 25-word sentences, 200-word paragraphs, 3 to 5 sections of 4 to 5 paragraphs, opener, conclusion, teaser) as the default the style guide may override. |
| `generator_coherence_instructions.txt` | The system prompt for the coherence pass. |
| `toc_project_instructions.md` | How to design a table of contents: the elicitation questions, the research step, the shape, the quality bar for each field. The model reads it before every book. |
| `toc_template.txt` | A short paste-in prompt for generating a TOC in a chat window, for people not running the model in this folder. |
| `config.example.yaml` | Every install setting, documented. Copied to `config.yaml`, which is git-ignored. |
| `style_guides/` | `TEMPLATE.txt`, the questionnaire's fill-in form with the readability skeleton fixed; four finished voices (generalist, kid-and-parent, evidence-first, science-explainer); `default.txt`, the install's own, git-ignored. |
| `examples/` | `test_toc.yaml` and `test_style_guide.txt`, the three-chapter first-run book; `wastewater_toc.yaml` and `wastewater_style_guide.txt`, a complete real outline from a finished book, the format reference. |
| `CLAUDE.md`, `.claude/skills/` | The entry file and the two Claude Code skills, `setup-bookwriter` and `write-a-book`. |
| `tocs/` | One folder per book designed here: brief, research, outline, style guide, logs. Git-ignored; it is the owner's content. |
| `requirements.txt` | `anthropic`, `ebooklib`, `beautifulsoup4`, `lxml`, `pyyaml`, `pymupdf`. |
| `GENERATOR_PLAN.md`, `DEVELOPMENT_HISTORY.md`, `plan.md` | The original design and the build history. |

### Command reference
```bash
python generator.py <toc.yaml> <style_guide.txt> [options]
```
| Flag | What it does |
|---|---|
| `--dry-run` | Validate the outline, print the chapter list, estimate the cost. No API calls. Always first. |
| `--research FILE` | Research notes, fed to every chapter as factual ground truth and archived into the book folder as `research_notes.md`. Use for anything recent or niche. |
| `--yes` | Skip the "Proceed with API calls?" prompt. For unattended runs, only after the person approved the dry run in conversation. |
| `--out DIR` | Put the book under this folder, overriding `output_dir` in `config.yaml`. |
| `--stage N` | Start from stage N (1 to 5). Resume after a failure, or re-run the coherence pass alone with `--stage 4`. |
| `--regen N` | Delete chapter N's cached files and rewrite it, then continue. Follow with `--stage 4`. |
| `--target N` | Override every chapter's word target. |

### The five stages
1. **Parse and validate.** Load both files, check required fields, create the book folder, flatten the hierarchy, print the plan and the estimate, confirm.
2. **Generate.** Each chapter in order with the generation model and adaptive thinking; each sees the outline, prior summaries (made cheaply with the summary model), the previous chapter in full, the next chapter's description, and its own detailed structure. Cached as HTML; re-runs skip finished chapters. The model pins are checked against the API before the first call.
3. **Validate.** Every chapter present; word counts against targets.
4. **Coherence pass.** Overlapping five-chapter windows, run concurrently, editing for transitions, redundancy, terminology, cross-references, arc, and voice. Books of seven chapters or fewer go in one window.
5. **Assemble.** EPUB with a nested table of contents and part title pages, plus one combined HTML file, in the book's `output/` folder.

### Settings (`config.yaml`)
`output_dir` (where books go; relative to this folder), `venv` (the environment the model should run; keep it outside synced folders), `default_style_guide`, `spend_ceiling` (the dry run warns above it), `reading_device`, `house_rules` (appended to the default style guide at setup). All optional. See `config.example.yaml`.

### Known issues
- **No fact-check stage.** Wrong facts read authoritatively. The README says so and the model offers a manual pass. A `checker.py` stage 6 (finished chapters plus research notes in, unsupported claims out) has been proposed since 2026-07-11 and not built.
- **Word counts run long**, ten to forty percent over target on the books so far. The default target is 1,300 so chapters land near the 1,500-word cap; the dry run warns about any chapter targeted above the cap.
- **The estimate runs high**; on the largest book it was more than double the bill. Treat it as a ceiling.
- **Model pins age.** The three constants at the top of `generator.py` name specific models. When one is retired the script stops before spending and says so. Changing a pin changes the book's voice, so it is the owner's decision, not an automatic fix.
- **A virtual environment inside a synced folder can hang** while the sync client fetches package files one at a time. Keep it outside, and record its path in `config.yaml`.
- **Coherence-pass damage** happens occasionally (a dropped paragraph, a flattened transition). Spot-read before delivering.
- **Non-fiction only.** A fiction mode would need its own instruction files.

### House style versus shared
Anything that names a person, a path, a device, or a taste lives in `config.yaml` or `style_guides/default.txt`, both git-ignored. The shipped style guides, the examples, and this README carry no one's paths and no one's rules. A house rule ("no em-dashes") is offered in the setup interview and appended to the default guide if chosen; it is never in the shared instruction files.

### Changelog habit
Record every change to the instruction files or the pipeline in `DEVELOPMENT_HISTORY.md` with the date and the reason. The prose voice depends on the instruction files, so a silent edit there changes every book after it.
