# Stanford CS329A — Self-Improving AI Agents

An independent, transcript-grounded textbook companion to **Stanford CS329A: Self-Improving AI Agents (Autumn 2025)**. The source course is taught by **Aakanksha Chowdhery and Azalia Mirhoseini**; the recordings are published by **Stanford Online**. These study notes are independently prepared and are not an official Stanford publication or endorsement.

**[Read the live textbook](https://az9713.github.io/cs329a-self-improving-agents/)** · **[Watch the source playlist](https://www.youtube.com/playlist?list=PLangBM27OtEA)** · **[Official Stanford course](https://cs329a.stanford.edu/)**

[![Preview of the CS329A textbook homepage — click to open the live website](https://az9713.github.io/cs329a-self-improving-agents/assets/preview.png)](https://az9713.github.io/cs329a-self-improving-agents/)

The preview and chapter links open fully rendered web pages on GitHub Pages, with typeset mathematics, recording timestamps, navigation, and interactive exercises. GitHub README pages cannot embed the site's live HTML or JavaScript.

## Read the chapters and watch the lectures

The chapter order follows the nine recordings in the supplied playlist. Chapter headings are editorial teaching titles; the video links lead to the corresponding original recordings. The [first source video](https://www.youtube.com/watch?v=6YnLB0XbTnI&list=PLangBM27OtEA) introduces the course.

| Part | Live textbook chapter | Stanford Online source video |
| --- | --- | --- |
| 01 | [Foundations and course overview](https://az9713.github.io/cs329a-self-improving-agents/lecture-01-overview.html) | [Watch Part 1](https://www.youtube.com/watch?v=6YnLB0XbTnI) |
| 02 | [Test-time compute scaling](https://az9713.github.io/cs329a-self-improving-agents/lecture-02-test-time-compute.html) | [Watch Part 2](https://www.youtube.com/watch?v=-Ggc37xLj_Y) |
| 03 | [Robust verification](https://az9713.github.io/cs329a-self-improving-agents/lecture-03-verification.html) | [Watch Part 3](https://www.youtube.com/watch?v=p7TdPUcPoik) |
| 04 | [Learning from feedback with tools and code](https://az9713.github.io/cs329a-self-improving-agents/lecture-04-feedback.html) | [Watch Part 4](https://www.youtube.com/watch?v=Lxh9RF5S-K0) |
| 05 | [Multi-step reasoning and planning](https://az9713.github.io/cs329a-self-improving-agents/lecture-05-planning.html) | [Watch Part 5](https://www.youtube.com/watch?v=Ml_fp9XkB8Y) |
| 06 | [Training and reinforcement learning](https://az9713.github.io/cs329a-self-improving-agents/lecture-06-training.html) | [Watch Part 6](https://www.youtube.com/watch?v=yVnmHSAy3ck) |
| 07 | [Self-improvement and deep research agents](https://az9713.github.io/cs329a-self-improving-agents/lecture-07-research-agents.html) | [Watch Part 7](https://www.youtube.com/watch?v=Uni9dqyuuDM) |
| 08 | [Agentic evaluation and long-horizon tasks](https://az9713.github.io/cs329a-self-improving-agents/lecture-08-evaluation.html) | [Watch Part 8](https://www.youtube.com/watch?v=8JAqLnTaZu4) |
| 09 | [Future research areas](https://az9713.github.io/cs329a-self-improving-agents/lecture-09-frontiers.html) | [Watch Part 9](https://www.youtube.com/watch?v=AyO6wyu4DEg) |

**Study tools:** [Interactive numerical lab](https://az9713.github.io/cs329a-self-improving-agents/numerical-lab.html) · [Timestamped source map](https://az9713.github.io/cs329a-self-improving-agents/source-map.html) · [Primary-source bibliography](https://az9713.github.io/cs329a-self-improving-agents/references.html)

## Our contribution beyond the transcripts

Stanford's lectures provide the topic sequence and intellectual starting point. Our contribution is the teaching edition built around that material:

- **Self-contained mathematical explanations.** We introduce notation, assumptions, definitions, derivations, and limiting cases locally, including the distinction between sampling coverage and delivered accuracy, verifier precision, policy-dependent values, reinforcement-learning objectives, and reliability over long task horizons.
- **Original worked teaching examples.** A recurring median-repair problem connects generation, checking, feedback, planning, learning, and evaluation. Numerical examples and counterexamples expose when an appealing conclusion fails.
- **Research synthesis and extensions.** We connect the lectures to primary papers and technical reports, distinguish a source's reported results from our deductions, and develop implications for building and evaluating agents. The underlying algorithms and research findings remain credited to their original authors; our contribution is the exposition, synthesis, and teaching constructions.
- **Practice with full reasoning.** Each chapter includes five exercises with worked solutions: **45 exercises** across the nine chapters, plus project directions and conceptual checks.
- **Executable and interactive learning.** The edition includes **16 executable Python examples**, a standard-library numerical verification script, and an interactive lab for sampling coverage, verifier precision, and task-horizon sensitivity. These illustrate simplified analytical models; they are not new empirical benchmark results.
- **A usable reading edition.** The HTML includes **47 timestamped main sections**, a searchable chapter index, linked primary references, mobile layouts, code-copy controls, printable solutions, and locally bundled mathematics for offline reading.

The complete edition contains approximately **32,000 words** across nine chapters and **34 primary references**. Word counts include code and mathematical markup; they are a size indicator, not a measure of lecture coverage.

## Source coverage and limitations

This edition covers the **nine playlist recordings**, not every meeting on the official Stanford syllabus. The syllabus also lists guest lectures and other sessions that are not reconstructed here.

Complete available English caption tracks were acquired. Editorial review used excerpts distributed across each recording and targeted passages around key topics. Slides and board images were **not comprehensively audited**. The notes are selective advanced explanations, not a verbatim transcript, a claim that every utterance was checked, or a replacement for the recordings. Caption-track metadata does not guarantee transcription accuracy.

Recording timestamps establish topic provenance and provide navigation. They do not imply that every adjacent derivation, example, or extension was spoken in the lecture. Chapter source notes and the [source map](https://az9713.github.io/cs329a-self-improving-agents/source-map.html) identify these boundaries. Historical benchmark comparisons should be read in their original context, not as current product rankings.

## Use locally

Download or clone this repository, then open `index.html` in a browser. The HTML, styles, mathematics, and numerical lab work offline. You need an internet connection only to follow recording and external research links. Use your browser's print command to print a chapter or save a PDF; exercise solutions expand for printing.

Each chapter also has an editable Markdown companion. To run the deterministic numerical checks with Python 3:

```bash
python numerical_lab.py
```

The script uses only the standard library and writes its results to `validation/numerical-checks.json`. It makes no API calls and requires no credentials. The interactive browser lab uses JavaScript and requires no Python installation.

## Verification and attribution

Before publication, the study edition passed checks covering the embedded Python examples, 25 numerical examples and boundary cases, local links, mathematics rendering, desktop and mobile layouts, and reading controls. A compact record is included in [validation-summary.json](https://github.com/az9713/cs329a-self-improving-agents/blob/main/validation-summary.json). These checks validate the reading artifact and computations; they do not establish exhaustive lecture coverage or reproduce research-paper experiments.

Credit for the course and recordings belongs to Stanford, the instructors, guest speakers, and Stanford Online. Credit for referenced research belongs to its authors. Our independently authored explanations, derivations, examples, exercises, and presentation are distinguished from those source materials. Raw caption files, recordings, private workspace records, and unrelated drafts are not redistributed in this repository. MathJax is bundled under its [Apache 2.0 license](https://github.com/az9713/cs329a-self-improving-agents/blob/main/assets/mathjax-LICENSE); that license applies to MathJax, not automatically to other repository content.

Prepared 25 September 2026. Course attribution: [Stanford CS329A](https://cs329a.stanford.edu/). Recording provenance: [Stanford Online playlist](https://www.youtube.com/playlist?list=PLangBM27OtEA).
