# Plan

Written before I started building, kept here as a record of what I intended and why.
Where the finished system differs, I've noted it.

## The task

Build a document-understanding system for semi-structured healthcare documents three
different ways — traditional NLP, a fine-tuned small model, and a general LLM — then
answer: what's the cheapest system that's good enough, and where does an LLM actually
earn its cost?

## What I decided up front

### Where the documents come from

There is no public corpus of labelled healthcare faxes, and real ones are PHI. So I
generate them: 6 document types across 14 layout templates, rendered to PDF, with real
ICD-10 and CPT codes and NPIs that pass the Luhn checksum.

The obvious risk is that synthetic data makes everything look easy. Two controls:

- **Templates 1–10 for training, 11–14 for the test set only.** Different geometry,
  different label wording. A model that memorises layout will visibly fall over.
- **Half the corpus goes through real OCR damage** — rasterise at 200 dpi, add skew,
  blur, speckle and JPEG artefacts, then read it back with an OCR engine. That noise
  isn't simulated, it happens.

I also planted two traps: a decoy 10-digit number that fails the NPI checksum, and a
`Printed:` footer date a few days after the real document date.

### The LLM problem

I have 7.4 GB of RAM and no API budget, which caps a local model at about 3B. A 3B
model is arguably itself a "small model", so calling it *the* general LLM would make the
central question unanswerable.

So two LLM tiers: Qwen2.5-3B running locally as the reproducible baseline, plus a
frontier tier annotated through Claude Code (no key, no spend) as a reference point. The
frontier tier's cost is estimated from published pricing, never metered, and labelled
that way everywhere it appears.

### Gold data

The assignment is explicit that the test set must be manually verified and must not come
from an LLM. My plan: the generator knows the true values, so most of the work is
confirming OCR didn't destroy them. Only the disagreements need a human.

*(Built as planned. 865 of 945 values agreed mechanically; I reviewed the other 80 by
hand.)*

### Silver data

Train the small model on labels the local LLM produced, never on the generator's truth —
that's what the assignment asks to demonstrate, and it's what you'd actually have in
production. Since the generator *does* know the truth, I can score the teacher, which
most silver pipelines never bother to do.

## Build order

| | |
|---|---|
| 1 | Generator, templates, OCR pipeline |
| 2 | Silver labelling + span alignment |
| 3 | Rules approach |
| 4 | Train the small model |
| 5 | Gold verification |
| 6 | Evaluation, error analysis, report |

Silver labelling and training both need the GPU and can't overlap — 4 GB of VRAM won't
hold Ollama and torch at once. Everything caches per document so an interrupted run
resumes.

## What I expected to find

That rules would be competitive on structured layouts and fail on prose; that the small
model would land between rules and the LLM; and that most of the remaining error would
turn out to be OCR damage rather than anything a model could fix.

Two of those held. The small model came in *below* rules — see the report for why, and
for the learning curve that explains it.

## Where the build diverged from this plan

- **1,300 documents, not 1,400**, and a **90-document** gold set rather than 120. The
  two-tier verification made 90 enough.
- **No real public documents in the gold set.** I couldn't find filled healthcare forms
  that were both public and label-able in the time available, so the gold set is entirely
  synthetic held-out templates. This is the main limitation of the whole project and the
  report says so.
- **torch cu128, not cu124** — matched the installed driver.
- I added a **learning curve** (100/300/600/1000 silver docs) that wasn't in the original
  plan. It turned out to be the measurement that explains the small model's result.
