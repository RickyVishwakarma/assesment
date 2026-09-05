# Document Intelligence: rules vs a small model vs LLMs

Four systems extract 15 fields and a document type from semi-structured healthcare
documents. All four run on the same 90-document test set, built from layout templates none
of them was trained or tuned on.

Every number here is measured and reproducible from the commands in the README. The one
exception is the frontier tier's cost per document, estimated from published per-token
pricing and labelled as such wherever it appears.

---

## The short version

| approach | F1 | accuracy | doc type | service lines | latency | size | cost/doc | offline | deterministic | hallucinates |
|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.825 | 0.705 | 0.911 | **0.000** | 11 ms | 6 MB | $0 | yes | yes | no |
| small model | 0.782 | 0.647 | 0.800 | **0.013** | 52 ms | 525 MB | $0 | yes | yes | no |
| local LLM 3B | 0.842 | 0.728 | 0.833 | 0.701 | 8,385 ms | 1.9 GB | $0 | yes | no | 0.006 |
| frontier LLM | 0.960 | 0.923 | 1.000 | 0.917 | — | — | $0.0067 | no | no | 0.005 |

Cost is $0 for the three local tiers because they run on hardware you already own; the
frontier figure is estimated from published pricing, never metered.

**Rules are the right default for flat fields.** 86% of frontier accuracy at 6 MB and 11 ms,
deterministically, with no ability to invent a value.

**But rules cannot do relationships at all.** On the service-line task — repeating rows of
`{procedure, date, units, charge, paid}` — rules score **0.000** and the small model 0.013,
against 0.701 and 0.917 for the two LLMs. That is not a gap that closes with tuning, and
section 6 is about why.

**A 3B local LLM is the worst option in the table.** It buys +0.017 F1 over rules — inside
the confidence intervals, so statistically a tie — for 645× the latency and 317× the size,
while adding non-determinism and hallucination risk. If you can't afford a frontier model,
the fallback is rules, not a small LLM.

**A third of the remaining error isn't a model problem.** `OCR_CORRUPTION` accounts for
exactly 51 errors for *every one* of the four approaches. Better scanning would move
accuracy more than any model change.

**The LLM earns its cost in three specific places**, none of which is reading characters
more accurately: telling apart two values with the same format, reading prose with no
labels, and handling layouts nobody wrote a rule for.

---

## 1. The test set

90 documents, 923 field values, all human-verified.

Values were resolved in two tiers. For 865 of 945 (91.5%) the generator's value and an
independent OCR pass agree — two processes produced the same string, so there's nothing to
adjudicate. The other 80 are cases where OCR and the generator disagree, and a person
decided each one against the page.

Of those 80: 22 excluded as destroyed past recovery, 1 corrected to what the document
actually says (`gold_synth-0042` reads `07-Sep-1982`, not the generator's `07-Sep-1952`),
the rest kept. Claude proposed decisions to speed the pass, and the reviewer overrode 22 of
them — so it was a real review, not a rubber stamp.

### The ceiling everything sits under

Half the corpus is rasterised at 200 dpi, degraded, and re-read through OCR. That destroys
information before any extractor runs:

| condition | gold values still present in the text |
|---|---|
| clean | 98.9% (430/435) |
| scanned | **85.3%** (435/510) |
| overall | 91.5% |

No extractive system can beat 85.3% on the scanned half. Read every scanned-condition
number against that, not against 100%.

---

## 2. The teacher

The small model trains on silver labels — what the local 3B model produced — never on the
generator's truth. Because the generator *does* know the truth, I can score the teacher,
which most silver pipelines skip.

| | |
|---|---|
| teacher field accuracy | 86.0% |
| teacher doc-type accuracy | 80.8% |
| span alignment rate | 96.6% |
| documents labelled | 1,200 (1,000 train / 200 val), ~3.5 h unattended, $0 |

Getting from a returned *value* to a trainable *character span* is the unglamorous step
that decides how much supervision survives:

| method | spans |
|---|---|
| exact | 11,914 |
| whitespace-flexible | 625 |
| compact (punctuation-insensitive) | 66 |
| fuzzy | 20 |
| unalignable | 165 |

Those 625 whitespace-flexible matches are supervision a naive `text.find(value)` would have
thrown away — values the layout broke across a line. About 5% of the training signal,
recovered by one tolerance decision.

### The teacher has a systematic bias, not just noise

`document_date` is the teacher's worst field at 62.7%, and the errors aren't random.
**106 of 110 are the model taking the `Printed:` footer date instead of the document
date** — always a day or three later:

| gold | teacher returned |
|---|---|
| `2026-02-10` | `2026-02-13` |
| `12/21/25` | `12/22/25` |
| `1/2/26` | `1/4/26` |

Every generated document carries that footer as a deliberate distractor. The teacher falls
for it almost every time.

This matters more than the headline accuracy:

> Random teacher noise gets averaged away by the student. Systematic teacher bias gets
> learned.

Section 4 tests that prediction.

---

## 3. How much silver data does the small model need?

Trained from the pretrained checkpoint at each size, identical hyperparameters, same test
set. 1,000 is the whole silver training set, so the last point is the shipped model — which
is why it reproduces the 0.782 in the headline table rather than merely approaching it:

| silver documents | F1 | gain |
|---|---|---|
| 100 | 0.184 | — |
| 300 | 0.623 | +0.439 |
| 600 | 0.721 | +0.098 |
| 1000 | 0.782 | +0.061 |

**100 documents is unusable.** There isn't enough signal to learn 31 BIO tags. Any
small-model result quoted from ~100 examples is noise.

**300 is the elbow** — 80% of the final F1 for 30% of the labelling.

**It's still climbing at 1,000.** The last step gains +0.061, so more silver data would
still help. But the teacher is only 86.0% accurate, so the curve will flatten near that
ceiling regardless of volume. Past this point the higher-leverage move is a better teacher,
not more documents from the same one.

---

## 4. Does the student beat its teacher?

Both run over the same 90 documents, so this is like-for-like.

The student loses overall — 0.782 against 0.842 — but the per-field picture is where the
content is.

| field | student | teacher | delta | train spans |
|---|---|---|---|---|
| `procedure_code` | 0.879 | 0.797 | +0.083 | 557 |
| `document_reference` | 0.895 | 0.824 | +0.072 | 796 |
| `document_date` | 0.372 | 0.324 | +0.048 | 889 |
| `patient_name` | 0.938 | 0.927 | +0.010 | 982 |
| `member_id` | 0.857 | 0.890 | -0.033 | 832 |
| `diagnosis_code` | 0.893 | 0.927 | -0.034 | 676 |
| `patient_dob` | 0.900 | 0.938 | -0.038 | 858 |
| `payer_name` | 0.946 | 0.987 | -0.041 | 839 |
| `total_charge` | 0.907 | 1.000 | -0.093 | 510 |
| `servicing_facility` | 0.703 | 0.828 | **-0.125** | 712 |
| `referring_provider_npi` | 0.808 | 0.948 | **-0.141** | 654 |
| `referring_provider_name` | 0.762 | 0.960 | **-0.198** | 921 |
| `date_of_service` | 0.460 | 0.723 | **-0.263** | 961 |
| `patient_responsibility` | 0.435 | 0.897 | **-0.462** | 160 |
| `amount_paid` | 0.348 | 0.867 | **-0.519** | 165 |

Three separate things are going on.

### Rare fields are catastrophic, and they carry the whole gap

| training spans | mean delta vs teacher |
|---|---|
| under 400 (`amount_paid`, `patient_responsibility`) | **-0.490** |
| 400 or more (the other 13) | **-0.058** |

Those two fields only appear on remittance advice, so they show up in ~160 of 1,000
training documents. The LLM handles them at 0.867 because it never needed examples.

Everywhere the student has adequate supervision it's roughly at parity with its teacher.
The aggregate gap is carried almost entirely by two rare fields — which makes the fix
class-balanced sampling, not a bigger model.

### The student inherited the teacher's bias, as predicted

`document_date` is where the teacher takes the `Printed:` footer 106 times out of 110. The
student scores 0.372 there, against 0.87–0.94 on comparable fields. It didn't correct the
bias; it reproduced it.

It technically edges the teacher (+0.048), so the prediction wasn't that the student would
score lower — it's that both are catastrophically wrong on the same field in the same
direction. Noisy labels are survivable. Consistently wrong labels are not.

### Same-format fields need reasoning the tagger doesn't have

`date_of_service` loses 0.263 despite 961 training spans, which breaks the supervision
explanation. Running the student over the test set and checking what it actually returned:
**11 of 60 `date_of_service` cases return `document_date`'s true value.**

`patient_dob` is fine (0.900) because a 1950s date is obviously a birth date. But
`document_date` and `date_of_service` are both recent dates in identical formats —
separating them means reading a label several tokens away. A 512-token tagger with no
layout signal is weak at exactly that. An LLM isn't.

**This is the clearest place a general LLM earns its cost:** not reading characters better,
but assigning the right role to two values that look the same.

---

## 5. Error analysis

1,004 errors coded across all four approaches on the same 90 documents.

| cause | rules | small model | local LLM | frontier |
|---|---|---|---|---|
| `FORMAT_NORMALISATION` | 41 | 78 | 45 | 90 |
| `OCR_CORRUPTION` | **51** | **51** | **51** | **51** |
| `MISSED_ENTIRELY` | 77 | 88 | 24 | **1** |
| `WRONG_VALUE` | 47 | 66 | 69 | 6 |
| `SPURIOUS_EXTRACTION` | 12 | 8 | 17 | 18 |
| `LABEL_CONFUSION` | 16 | 18 | 21 | **0** |
| `OVER_CAPTURE` | 19 | 5 | 4 | 0 |
| `TRUNCATION` | 7 | **18** | 3 | 0 |
| `HALLUCINATION` | 0 | 0 | 2 | 0 |
| **total** | **270** | **332** | **236** | **166** |

### The number that should drive the roadmap

`OCR_CORRUPTION` is **51 for all four approaches**. Not similar — identical. The same 51
field decisions fail everywhere, because the value was destroyed before any extractor ran.

That's 31% of the frontier model's entire error budget, and it's completely insensitive to
model choice. You cannot buy your way out of it with a better model, only with better
capture.

### What each approach's profile says

**Rules fail by not finding things, and almost only on one layout.** 77 `MISSED_ENTIRELY`,
of which **73 are on prose templates** — against 3 on grid and 1 on two-column. A dictionary
of label anchors has nothing to match when the value sits inside a sentence. That's not
general weakness; it's one precisely located blind spot.

**The small model fails at span boundaries.** It has by far the most `TRUNCATION` (18) and
the failures are diagnostic:

| gold | returned |
|---|---|
| `Riverbend Cardiology Institute` | `Riverben` |
| `84443` | `84` |
| `6/10/52` | `/10` |

Those are subword tokenisation boundaries, not comprehension failures — the model tags the
first WordPiece and drops the continuation. A decoding fix (enforce whole-word spans when
merging BIO tags) worth maybe 2–3 F1 points, not evidence the architecture is wrong.

**The frontier model barely misses anything (1) but reformats the most (90).** Its dominant
error is returning a value in a different surface form than the document used. The
asymmetry is worth noting: the frontier tier's errors are mostly cosmetic, the small
model's are mostly structural.

One caveat on that 90, found while sampling it: a share of those rows are cases like
`James Nguyen` returned as `James   Nguyen` — doubled spacing the OCR introduced and the
model copied faithfully. The classifier codes that as `FORMAT_NORMALISATION` because the
strings differ in whitespace, but the cause is scan damage. So `OCR_CORRUPTION` at 51 is a
floor on the OCR share of error, not a full accounting, and the normaliser should collapse
internal whitespace before comparison.

**`LABEL_CONFUSION` is 0 for the frontier model and 16–21 for everything else.** Section 4's
finding arriving from the other direction.

### Sliced by scan condition and language

One overall score hides the effects that matter. Every approach was also scored by OCR
condition and by document language:

| approach | clean | scanned | drop | English | Spanish |
|---|---|---|---|---|---|
| rules | 0.853 | 0.802 | −0.051 | 0.823 | 0.875 |
| small model | 0.813 | 0.755 | −0.058 | 0.782 | 0.778 |
| local LLM 3B | 0.890 | 0.800 | −0.090 | 0.843 | 0.825 |
| frontier LLM | 1.000 | 0.924 | −0.076 | 0.958 | 1.000 |

**Scanning costs everyone 5–9 points**, and it costs the LLMs *more* than the rules engine,
not less. A regex either matches the damaged string or doesn't; a language model reads a
corrupted token and confidently produces a plausible wrong value.

**Spanish is not a weak slice, which surprised me.** I expected multilingual to be a visible
error category and built it into the corpus (13% of documents) expecting to report a
penalty. Rules score *higher* on Spanish (0.875 vs 0.823) and the frontier tier scores 1.000
on it. The reason is mundane: the fields that carry the signal — NPIs, ICD-10 and CPT codes,
dates, amounts — are language-independent, and the Spanish templates use consistent label
wording that the dictionary picked up from training documents. The honest conclusion is that
this corpus does not test multilingual robustness, not that multilingual extraction is easy.
Real Spanish clinical documents would not be a translation of an English template.

### Twenty-two representative errors and their fixes

| approach | field | gold | predicted | cause | fix |
|---|---|---|---|---|---|
| rules | `patient_name` | `William Davis` | — | prose has no anchor | NER fallback for unanchored layouts |
| rules | `member_id` | `NST595159366` | `NST595159366.` | sentence period captured | strip trailing punctuation on ID types |
| rules | `document_reference` | `AUTH-0274973` | `AUTH-0274973 was` | regex ran past the value | anchor to token boundary |
| rules | `date_of_service` | `January 29, 2026` | `March 14, 2026` | took the issue date | require label proximity, not first-date |
| rules | `patient_responsibility` | `$79.25` | `2` | grabbed the units column | type-check amount candidates |
| rules | `patient_name` | `Steven Martin` | `INTAKE FORM` | matched the title | exclude all-caps header lines |
| small | `servicing_facility` | `Riverbend Cardiology Institute` | `Riverben` | subword boundary | whole-word span merge |
| small | `procedure_code` | `84443` | `84` | subword boundary | as above |
| small | `date_of_service` | `03.12.2025` | `04.03.2025` | same-format date confusion | label-proximity feature |
| small | `member_id` | `YCF591565308` | `7770346428` | returned the NPI instead | checksum-reject NPIs for ID fields |
| small | `document_reference` | `LAB6478780` | — | scanned `below` layout, tag never fired | more scanned examples in silver |
| small | `payer_name` | `Ambetter Health` | `Ambetter  Health` | doubled space kept from OCR | collapse whitespace before comparison |
| local LLM | `document_date` | `5/9/25` | `5/12/25` | `Printed:` footer bias | one line in the prompt |
| local LLM | `date_of_service` | `03.12.2025` | `04.03.2025` | swapped the two dates outright | name both fields in the prompt with their anchors |
| local LLM | `procedure_code` | `99213` | `84443, 73130, 80053` | invented a list — one of 2 hallucinations | constrain to one value; verify against source text |
| local LLM | `procedure_code` | `70450` | `70553, 70450, 99214` | right answer buried in invented siblings | as above |
| local LLM | service lines | 3 rows | 8 rows | recall 0.905 but precision 0.571 | de-duplicate rows; require a charge to appear verbatim |
| frontier | `diagnosis_code` | `E03.9` | `E03.9.` | trailing sentence period | strip terminal punctuation on code types |
| frontier | `patient_name` | `James Nguyen` | `James   Nguyen` | OCR spacing carried through | whitespace-collapse in the normaliser |
| frontier | `document_date` | `28-Aug-2025` | `2k-Aug-2025` | read the damaged glyph faithfully | nothing at model level — capture problem |
| frontier | `document_date` | `08/26/2025` | — | its single miss in 90 documents | none warranted |
| all four | `member_id` | `YCF591565308` | `Y.CF59 1565308` | OCR damage | better capture — no model fix exists |

That last row is the whole argument in one line: every approach returns the same corrupted
string, because that's genuinely what the document says after scanning.

---

## 6. Relationships: the service-line task

Flat fields are only half the problem the assignment describes. Claims and remittance
advices carry a *table* of service lines — repeating rows of
`{procedure_code, date_of_service, units, charge, paid}` — and getting one right means
getting five values right and keeping them on the same row. 30 of the 90 gold documents
carry them, 84 rows in total, scored as set-F1 over normalised tuples.

| approach | precision | recall | F1 | rows matched of 84 |
|---|---|---|---|---|
| rules | 0.000 | 0.000 | **0.000** | 0 |
| small model | 0.015 | 0.012 | **0.013** | 1 |
| local LLM 3B | 0.571 | 0.905 | 0.701 | 76 |
| frontier LLM | 0.917 | 0.917 | **0.917** | 77 |

This is the widest margin anywhere in the project, and it is a difference in kind rather
than degree.

**Rules score zero because the task has no anchor.** The whole approach keys on finding a
label and reading the value beside it. A service-line table has one header for N rows, so
there is nothing to match per row, and no way to express "these five values belong
together" in a dictionary of label variants. Adding more label patterns cannot move this
number off zero.

**The small model scores 0.013 for a related reason.** BIO tagging marks *which tokens are a
charge*; it has no representation for *which row a charge belongs to*. It recovered exactly
one complete row out of 84 — and that is the ceiling of the architecture, not of its
training data. Row grouping needs either a table-structure model (LayoutLM and relatives
encode 2-D position for exactly this) or a decoder that emits structured output.

**The LLMs handle it because emitting a JSON array is native to them.** Note the local
model's asymmetry: recall 0.905 against precision 0.571. It finds nearly every real row and
invents a lot of extra ones — 57 false positives. The frontier tier is balanced at
0.917/0.917.

So the honest form of the headline recommendation is narrower than "rules win":

> Rules are the cheapest adequate system **for flat field extraction**. For structured
> repeating content they don't compete, and no amount of rule-writing changes that.

If a production pipeline needs service lines — and a claims pipeline does — the routing
design in section 8 stops being a cost optimisation and becomes a requirement: flat fields
by rules, tables by an LLM.

---

## 7. Determinism

Each approach run over the same 5 documents 3 times; the figure is the fraction whose
output was byte-identical every time.

| approach | rate | what it means |
|---|---|---|
| rules | 1.00 | deterministic by construction |
| small model | 1.00 | argmax over logits, no sampling |
| local LLM 3B | 1.00 on 5 docs | but 3 full 90-doc passes scored 0.835 / 0.841 / 0.842 |
| frontier LLM | 1.00 | artefact of the cache, not a property of the model |

Only the first two are claims. The rules engine is a program and the tagger takes an argmax
in eval mode — their 1.00 confirms nothing non-deterministic crept in.

The local LLM's 1.00 is not just weaker than it looks — at 90 documents it stops holding.
Three full evaluation passes over the identical gold set scored **0.835, 0.841 and 0.842**,
with the hallucination rate moving between 0.005 and 0.007. So the 5-document result says
only that divergence is rare, not that it doesn't happen; scaled to 90 documents it appears
reliably. Greedy decoding is deterministic given identical numerics, but batching, GPU
non-associativity and server-side cache reuse can each change a tie-break, and one changed
token is one changed field.

That is the practical form of the problem. A 3B local LLM cannot be audited by re-running
it, because re-running it does not reproduce the previous answer — and the resulting spread
(0.007) is wide enough to swallow its entire measured lead over the rules engine.

The frontier tier's 1.00 says nothing at all — it's replayed from a cache. A live frontier
API is the one tier here that genuinely can't promise reproducibility, since model versions
change underneath you.

For a clinical pipeline what matters isn't the rate but the reason: a
deterministic-by-construction system can be audited by re-running it. A sampled one can't,
however stable it looks on five documents.

---

## 8. The answer

> *What's the smallest/cheapest system that achieves acceptable accuracy, and where does a
> general LLM still provide meaningful value?*

### Rules are the smallest adequate system

86% of frontier accuracy at 6 MB, 11 ms and zero marginal cost. Nothing else in the table
is close on accuracy per unit of cost.

The comparison that should actually decide the architecture is rules against the local 3B:

| | rules | local LLM 3B | |
|---|---|---|---|
| F1 | 0.825 | 0.842 | LLM ahead by +0.017 |
| latency | **11 ms** | 8,385 ms | **762× slower** |
| size | **6 MB** | 1,900 MB | **317× larger** |
| deterministic | **yes** | no | |
| can hallucinate | **no** | yes (0.006) | |

A 3B model buys +0.017 F1 for 762× the latency, and that margin sits inside the bootstrap
confidence intervals — on this corpus the two are statistically indistinguishable.

That margin is not stable, in two separate ways. It moved during the project — before gold
verification rules were *ahead* by 0.002, and restoring 21 borderline fields flipped the
order. And it moves between runs: three full passes of the local LLM over the same 90
documents scored 0.835, 0.841 and 0.842, a spread of 0.007 that is itself larger than the
0.002 the adjudication turned on. **When two systems trade places depending on how you
adjudicate 21 of 923 fields — and one of them won't return the same score twice — the
accuracy difference isn't what should decide the architecture.** The latency gap, the
determinism and the inability to hallucinate are not close, and they don't move.

The fine-tuned small model doesn't justify its 525 MB either, scoring below the rules it was
meant to replace. Section 4 explains why, and the fix is cheap. Its structural advantage is
real but narrow: it cannot hallucinate, and unlike rules it degrades gracefully on layouts
nobody wrote a dictionary for.

### Most of the remaining gap isn't a model problem

Two measurements bound how much any model choice can buy. Only 85.3% of values survive OCR
on scanned documents; and `OCR_CORRUPTION` is 51 errors for all four approaches alike.

A large share of the headline gap is a scanning problem. Better capture — higher DPI,
deskewing, a stronger OCR engine — would move accuracy more per pound than upgrading the
model, and it helps every approach at once.

### Where the LLM genuinely earns its cost

Four places, each measured:

1. **Structured repeating content.** Service lines: rules 0.000, small model 0.013, local
   LLM 0.701, frontier 0.917 (section 6). This is the only task in the project where the
   non-LLM approaches don't merely score lower — they don't work at all.
2. **Roles for values that look identical.** The small model returns `document_date`'s value
   for `date_of_service` in 11 of 60 cases; the LLM scores 0.740 against the tagger's 0.460.
3. **Documents with no anchors.** 73 of the rules engine's 77 misses are on prose, where a
   value sits inside a sentence with no `Label:` to key on. Rules don't degrade here, they
   fail outright.
4. **Layouts and wording nobody anticipated.** The dictionary genuinely doesn't know five
   label variants that only occur on held-out templates. A dictionary knows what it was
   told; an LLM doesn't need telling.

### What I'd build

```
             +-- confident --> accept           (~85% of documents, 11 ms, $0)
document --> rules
             +-- not confident --> frontier LLM  (~15%, $0.0067)
```

Route on the conditions the measurements identify: prose layout, unfamiliar label
vocabulary, low field confidence, or a same-format field pair in play. On this corpus that
escalation costs roughly $10 per 10,000 documents rather than $67, while recovering most of
the 0.135 F1 gap.

**One caveat that matters for this design:** the confidence scores aren't calibrated. I ran
a real blank CMS-1500 from cms.gov through the rules engine and got garbage —
`patient_name = "Self Spouse Child Other"` — at confidence 0.90–1.05. Those scores reflect
which rule fired, not whether the answer is right. Routing needs abstention first: require a
label anchor within a bounded distance, type-validate the value, and reject fields whose
document-type prior doesn't match.

### What I'd do next, in order

1. **Fix the two rare fields.** ~160 training spans each, -0.490 mean F1 against the
   teacher. Class-balanced sampling, not a bigger model.
2. **Improve capture before improving models.** The 85.3% ceiling caps everything.
3. **Add abstention and calibrate confidence.** The routing design above doesn't work
   without it.
4. **Fix the teacher's `Printed:`-date bias** with one prompt instruction; the student
   inherits the improvement.
5. **Then** revisit the small model, which the learning curve says is still data-limited.

---

## Limitations

**The corpus is synthetic.** No public set of labelled healthcare faxes exists and real ones
are PHI. Real documents have handwriting, stamps and multi-page threading this generator
doesn't produce, so absolute numbers would drop on them. What I'd defend is the relative
comparison — identical documents, held-out layouts, real OCR damage.

**No real documents in the test set.** The original plan included a slice of public forms; I
couldn't find filled, labelable healthcare documents in the time available. This is the
single biggest gap.

**The frontier tier isn't independently reproducible.** It's a cached reference point, and
its cost is estimated rather than metered.

**Determinism is measured on 5 documents.** Enough to detect gross instability, not enough
to establish reproducibility — and at 90 documents the local LLM does diverge (section 7).

**The error taxonomy is assigned by rule, not by hand.** Causes are inferred from the gold
and predicted strings, which is consistent across approaches but imperfect: some whitespace
damage lands in `FORMAT_NORMALISATION` when it belongs in `OCR_CORRUPTION`. The 51/51/51/51
identity is exact; the boundaries between the softer categories are approximate.

**Spanish is a weak test of multilingual robustness.** The Spanish documents are template
translations, not independently authored clinical text (section 5).

---

## Appendix: leaks found and closed

Three bugs during the build would each have flattered a result. They're listed because a
comparison is only as trustworthy as its worst unexamined assumption.

1. **The rules approach imported the generator's own label dictionary**
   (`from ..gen.render import LABELS`), handing it perfect foreknowledge of wording that only
   appears on held-out templates. It now builds its vocabulary from training documents alone.
   Five variants (`PAID AMT`, `PT RESP`, `Payment`, `Saldo`, `You Owe`) are consequently
   unknown to it, and the resulting misses are a finding about dictionary brittleness rather
   than a defect to patch.

2. **The evaluator labelled all gold records `human_verified`** whether or not a human had
   reviewed them, keying on presence in the file rather than each record's own flag.

3. **OCR assembly collapsed two-column rows to a single space**, making
   `Patient Name:` / `Reference #:` indistinguishable from prose and merging adjacent
   columns' values into one field — `"Elizabeth Rivera : ORD-4322856"` as a patient name.
   Column gaps are now preserved, as `pdftotext -layout` does.

`tests/test_no_leakage.py` guards all three.
