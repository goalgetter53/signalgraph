# signalgraph

**Account intelligence from public buying signals.** signalgraph pulls funding filings, job
postings, engineering activity and news from public APIs. It resolves the messy company names in
those feeds into one canonical account per company and ranks the accounts with a score that
explains itself line by line.

> Python 3.11 · async httpx · DuckDB · FastAPI · numpy · hypothesis · ruff + mypy --strict clean ·
> 169 tests (167 offline + 2 opt-in live)

---

## 1. The problem

Outbound teams already know the best time to reach a company: right after it raises money, starts
hiring its first sales team, or ships a big release. That information is public. It is also spread
across sources that disagree about **who** the company is:

| Source | How the same company shows up |
|---|---|
| SEC Form D | `NORTHWIND ROBOTICS INC`, no website |
| Greenhouse | `Northwind Robotics`, `northwindrobotics.com/careers?gh_jid=…` |
| GitHub | `northwind-robotics` |
| A news feed | `Northwind Robotcs` (typo) |
| A teammate's watchlist | `Acornfield Labs`, a name the company dropped last year |

Many "signal" setups are a Clay table plus an n8n flow that joins these records on an exact name
or domain match. That join has two problems. It misses real matches: typos, rebrands, `Inc` vs
`Incorporated`, legal names with no website. It also creates false ones: `Northwind Robotics
GmbH` (the German subsidiary, with its own `.de` domain) is merged into the US parent, the US
and Indian "Summit Solutions" become one company, and two startups that list their incubator's
URL get merged with the incubator and with each other. On the held-out half (2,521 records) of a
real company gold set built from Wikidata, that naive join reaches **F1 0.282** (on my hand-made
synthetic gold set it reaches 0.784).
Every miss is a lost touchpoint, and every false merge sends a rep to the wrong company.

## 2. What I built

1. **Connectors** for SEC EDGAR Form D, Greenhouse, Lever, Ashby, GitHub org events and RSS/Atom.
   They are async and polite: per-host rate limits, retries with full jitter, `Retry-After`
   support, and a contact User-Agent. Each one normalises into a typed `SignalEvent`.
2. **An entity-resolution engine** built from canonicalisation, blocking, a learned pair scorer
   and constrained union-find. On the **held-out TEST half of a real Wikidata gold set** (2,521
   records, 1,108 companies) the current (v2) engine reaches **P 0.708 / R 0.509 / F1 0.592**,
   against **F1 0.393** for the v1 engine and 0.282 for the naive join (paired 95% CI of the
   v2 − v1 gain: +0.152 to +0.247; §4). That is still far below the F1 0.915 it scores on my
   173-record *synthetic* gold set (0.974 for v1), and the README explains why.
3. **Explainable scoring.** Each signal type has a weight and an exponential half-life, magnitude
   is log-scaled, and each type has a cap. ICP filters live in YAML. `GET /accounts/{id}/explain`
   returns every term of the score.
4. **Incremental, idempotent storage** in DuckDB. Each connector has a watermark and every upsert
   is keyed on a deterministic id, so running a step twice adds nothing (there is a test for it).
5. **Interfaces:** a CLI (`signalgraph ingest|resolve|score|export|serve|runs`), a read-only
   FastAPI service, and exports to CSV, Clay webhook rows and HubSpot batch-upsert JSON.

## 3. Architecture

```mermaid
flowchart LR
  subgraph Sources[Public sources]
    E[SEC EDGAR<br/>Form D]
    G[Greenhouse]
    L[Lever]
    A[Ashby]
    H[GitHub org events]
    R[RSS / Atom]
  end
  subgraph Connectors[connectors/ - async httpx]
    PC[PoliteClient<br/>rate limit · jittered retry · Retry-After · UA]
  end
  Sources --> PC --> SE[SignalEvent<br/>typed, normalised]
  SE -->|upsert on sha256 id<br/>advance watermark| DB[(DuckDB<br/>signals · mentions)]
  DB --> ER
  subgraph ER[resolution/ - entity resolution]
    N[canonicalise<br/>names · eTLD+1 · aliases] --> B[block<br/>domain · rarest token · prefix]
    B --> F[pair features<br/>JW · TF-IDF · residual · numbers · domain · country]
    F --> S[logistic regression<br/>P match]
    S --> U[union-find<br/>+ domain cannot-link]
  end
  ER -->|replace in txn| ACC[(accounts · members)]
  ACC --> SC[scoring/<br/>weights × magnitude × decay<br/>caps · ICP rules]
  SC -->|replace in txn| SCO[(scores + breakdowns)]
  SCO --> API[FastAPI<br/>/accounts · /signals · /explain]
  SCO --> EX[export<br/>CSV · Clay · HubSpot]
  SCO --> CLI[CLI]
```

The layers depend inward only. `domain/` holds pure models and has no I/O.
`connectors/`, `resolution/` and `scoring/` depend on `domain/`. `storage/` is the only module
that writes SQL. `services/` orchestrates the steps, and `api/` and `cli.py` are thin adapters.
Three ADRs in [`docs/adr/`](docs/adr) record the main decisions: the store, the ER design, and
the scoring model.

## 4. Entity resolution deep-dive

### Pipeline

| Stage | What it does | Why |
|---|---|---|
| **Name canonicalisation** | NFKC + casefold, strip accents, `&`/`+` → `and`, drop name-TLDs (`Orbitly.io`), collapse dotted abbreviations (`S.A.`, `K.K.`, `L.L.C.`), strip trailing legal forms in 15+ jurisdictions (`GmbH`, `Pvt Ltd`, `s.r.o.`, `Kabushiki Kaisha`…), extract `f/k/a` / `formerly` / `d/b/a` aliases | `Zentro Inc. (f/k/a Acornfield Labs, Inc.)` → `zentro` + alias `acornfield labs` |
| **Domain canonicalisation** | eTLD+1 via the Public Suffix List **including the private section**; unwrap Google/Facebook/LinkedIn/Outlook redirect URLs; emails → domain; IDN → punycode; free-mail, ATS and social hosts → *unknown*; tenant subdomains on Substack, Webflow and similar hosts are kept | `latticebloom.github.io` ≠ `pinehollow.github.io`, and `boards.greenhouse.io/x` says nothing about company x |
| **Blocking** | Candidate pair if two records share the canonical domain, one of the **3 rarest** name tokens (by document frequency in the batch), a 4-char prefix, or an **initialism** key (`IBM` ↔ `International Business Machines`). Oversized name blocks are skipped *and counted* | Avoids O(n²). (Throughput numbers in §6 were measured with v1's single rarest-token key) |
| **Features** (11) | Jaro-Winkler, token-set TF-IDF cosine, token containment, **residual similarity** (JW of the non-shared tokens), **number mismatch** (`Fund II` vs `Fund III`), domain match/mismatch, country match/mismatch, **site mismatch** (same registrable domain, different regional/section scope: `microsoft.com/en-ca` vs `microsoft.com/fr-ma`), **acronym match**. A missing value counts as neutral, not as disagreement | |
| **Scorer** | L2 logistic regression fitted with IRLS in numpy, on synthetic pairs + the candidate pairs of the real Wikidata **DEV** split (×10). Fellegi-Sunter is implemented as a baseline and compared | Weights ship as a readable JSON file ([`er_model.json`](src/signalgraph/resolution/er_model.json)) |
| **Clustering** | Union-find over edges (p ≥ 0.10), strongest first, with two **cannot-link** constraints: never merge clusters holding different **site keys** (domain + regional/section scope), and never merge clusters if any scored pair across them has p < 0.05 (**veto**) | Stops chaining through domain-less records and across subsidiary families |

### Results on real data (Wikidata companies)

#### v2: held-out TEST split, before vs after

Measured by `make real-eval` → [`reports/real_eval_v2.json`](reports/real_eval_v2.json) /
[`reports/real_eval_v2.md`](reports/real_eval_v2.md).

- **Split.** [`data/real/wikidata_split.csv`](data/real/wikidata_split.csv)
  ([`make_split.py`](benchmarks/real/make_split.py), seed 20261120) splits the 2,187 companies
  into DEV (1,079 companies, 2,523 records) and TEST (1,108 companies, 2,521 records) **by company
  group**: companies linked by a parent/subsidiary edge or by a shared normalised name are
  always on the same side, so no company and no family of hard negatives straddles the split.
- **DEV only** was used for error analysis, feature work, retraining and the choice of threshold,
  veto, blocking width and real-pair weight. The choice used 5-fold **out-of-fold** scores
  within DEV ([`tune_dev.py`](benchmarks/real/tune_dev.py) →
  [`reports/real_tune_dev.json`](reports/real_tune_dev.json)), so the threshold was not picked
  on pairs the model was fitted on. TEST was scored once at the end.
- **before** = the v1 engine exactly as shipped (commit `7ab183f`), frozen as clusters in
  [`reports/real_eval_v2_before_clusters.json`](reports/real_eval_v2_before_clusters.json) by
  [`predict_split.py`](benchmarks/real/predict_split.py). Each split is resolved as its own
  batch. CIs: paired percentile bootstrap (1,000 resamples) over connected components.

| TEST (held out): 2,521 records, 2,117 true pairs | Precision | Recall | **F1** | TP / FP / FN | 95% CI (F1) |
|---|---|---|---|---|---|
| Naive: exact normalised name OR same domain | 0.386 | 0.223 | 0.282 | 471 / 749 / 1646 | 0.234–0.338 |
| Engine before (v1, shipped) | 0.525 | 0.314 | 0.393 | 664 / 600 / 1453 | 0.335–0.452 |
| **Engine after (v2)** | **0.708** | **0.509** | **0.592** | **1077 / 444 / 1040** | **0.550–0.630** |

- **Paired difference on TEST:** after − before F1 **+0.152 to +0.247** (95% CI over 699
  components); after − naive +0.266 to +0.351.
- **Context only, not results:** on DEV (in-sample for v2) after/before F1 is 0.608 / 0.411; on
  the full 5,044 records (half of them DEV, so *partly in-sample*) it is **0.576** / 0.400
  (naive 0.298; the naive number moved from 0.291 because v2 normalises more legal forms).
- **DEV ablations** (out-of-fold, best threshold per row): all changes 0.601; without the veto
  0.580; synthetic-only training 0.511; one rarest-token key instead of three 0.577.

What changed (all general, nothing keyed to a company name):

1. **Site key with regional/section scope.** `microsoft.com/en-ca`, `nintendo.com/hk/` and
   `poland.arcelormittal.com` keep their scope (`domains.site_key`); generic hosts and paths
   (`www`, `jobs.`, `/careers`, pure language codes like `/de/`) do not. The cannot-link
   constraint now uses site keys, and a new `site_mismatch` feature marks "same domain,
   different section".
2. **Score veto in clustering.** Two clusters are not merged if any scored pair across them is
   below p = 0.05, which stops chains of individually plausible edges.
3. **Legal forms.** `Public Limited Company`, `Joint Stock Company`, `Tbk`, `Ltée`, `(publ)`,
   `S.A.B. de C.V.`, and leading `PT` / `OOO` are stripped.
4. **Blocking.** Keys on the 3 rarest tokens (was 1) plus an initialism key. Blocking pair
   completeness on the full set rose from 0.531 to 0.667 (0.658 on TEST).
5. **Acronym feature** and **retraining** on synthetic + real DEV pairs; threshold **0.10**
   (was 0.40), picked on DEV.

**Honest read.** v2 is clearly better on held-out real data: on TEST, sibling-subsidiary false
merges fell from 330 to 49 and parent/subsidiary ones from 173 to 125, and recall rose from 0.314
to 0.509. What still fails:

- **Unrelated false merges went up (97 → 270 on TEST).** The low threshold plus containment
  edges let a generic name chain into a family: `Italia` → `IBM Italia` → the whole IBM group.
- **No lexical overlap is still unsolved**: tickers (`INTC` for Intel), rebrands (`Facebook,
  Inc.` → `Meta Platforms`), and translated official names (`Freie und Hansestadt Hamburg`).
  These need an alias/ticker source, not a better string matcher. A third of TEST true pairs
  (34%) are still never proposed by blocking.
- **Regression on the synthetic gold set: F1 0.974 → 0.915** (P 0.939, R 0.891 at 0.10; see
  below). The real DEV data has exactly one website per company, so "same domain" is never a
  positive there and the `domain_match` weight fell from 3.96 to 1.64. One unit-test fixture
  shows the cost: `Acornfield Labs` (zentro.ai) no longer joins `Zentro` (zentro.ai) without a
  bridging alias. I kept the DEV-chosen settings rather than tune against the synthetic gold
  set, which is a test set too.

#### v1: full set, before any changes (historical)

Measured by `benchmarks/real/run_real_eval.py` at commit `7ab183f` →
[`reports/real_eval.json`](reports/real_eval.json) /
[`reports/real_eval.md`](reports/real_eval.md). Everything in this subsection describes the v1
engine. The data was pulled from the public Wikidata
SPARQL endpoint on **2026-11-20** by
[`benchmarks/real/build_wikidata_gold.py`](benchmarks/real/build_wikidata_gold.py) (`make
real-data`). Wikidata is **CC0 1.0**, so the derived gold set is committed:
[`data/real/wikidata_companies_gold.csv`](data/real/wikidata_companies_gold.csv).

- **Ground truth = Wikidata QID.** Each company becomes up to 4 records built from real strings
  only, with no injected typos: `careers` (English label + official website P856), `registry`
  (current official name P1448 + country, no website), `news` (one English alias, no website) and
  `watchlist` (a former official name, i.e. P1448 with an end date). That gives 2,187 careers,
  1,372 news, 998 registry and 487 watchlist records.
- **Sampling (hard cases on purpose):** 913 companies that are a parent or subsidiary of another
  sampled company (P749), 648 rebranded businesses, 513 generic businesses with a website and an
  official name, and 113 other companies whose English label exactly equals a sampled label.
  Parents, subsidiaries and same-name companies are separate entities, so they are hard negatives.
- **Protocol (v1): no fitting and no tuning on this data.** The v1 `er_model.json` (trained on
  synthetic data only) runs through the production `EntityResolver` at the shipped threshold
  0.40. Metrics are pairwise over all 12,718,446 record pairs. The CIs come from a percentile
  bootstrap (1,000 resamples) over the 1,520 connected components of truth ∪ predictions.

| Method @ 0.40, 5,044 records, 4,279 true pairs | Precision | Recall | **F1** | TP / FP / FN | 95% CI (F1) |
|---|---|---|---|---|---|
| Baseline: exact normalised name OR same domain | 0.422 | 0.222 | 0.291 | 950 / 1299 / 3329 | 0.263–0.320 |
| **signalgraph engine (v1)** | **0.528** | **0.322** | **0.400** | **1378 / 1231 / 2901** | **0.366–0.431** |

- **Paired difference:** engine − baseline F1 has a 95% CI of **+0.095 to +0.124**.
- **Blocking** proposed 12,847 candidate pairs but kept only **53.1%** of the true pairs. Nearly
  half of the true pairs share no blocking key: no domain, no rarest name token and no
  4-character prefix.
- **The threshold is not the problem.** Picking the threshold on this test set (cheating, for
  diagnosis only) gives a best F1 of 0.461, at threshold 0.10.

| Sampling slice (pairs within slice) | Records | Entities | Engine P / R / F1 | Baseline P / R / F1 |
|---|---|---|---|---|
| generic businesses | 1258 | 513 | 0.973 / 0.400 / 0.567 | 0.985 / 0.270 / 0.424 |
| rebranded businesses | 2001 | 648 | 0.912 / 0.289 / 0.439 | 0.918 / 0.202 / 0.331 |
| same-label companies | 183 | 113 | 0.714 / 0.309 / 0.431 | 0.518 / 0.346 / 0.415 |
| parents & subsidiaries | 1602 | 913 | 0.241 / 0.324 / 0.276 | 0.163 / 0.210 / 0.183 |

For the README's core use case, matching a legal name with no website (`registry`) to a brand
name with a website (`careers`), engine recall is 0.627 against the baseline's 0.501.

**Honest read.** The engine does **much worse on real data than on my synthetic set** (F1 0.400 vs
0.974). It still beats the naive join on F1 in every slice, and the paired CI on the gain excludes zero.
There are three reasons for the drop:

- **Many true pairs have no lexical overlap.** No string matcher can link these without an
  external alias or ticker table:
  - rebrands (`CBS RECORDS CANADA LTD.` → `Sony Music Canada`, `Merloni Elettrodomestici` →
    `Indesit Company`),
  - tickers and acronyms used as aliases (`BUMI` for `Bumi Resources`, `BOEV` for `Börsenverein
    des Deutschen Buchhandels`),
  - multilingual legal names.
- **The model is too cautious when one name adds words.** `Crown Resorts Limited` vs `Crown
  Limited` scores 0.288. The likely cause is that my synthetic generator teaches "same brand + different extra word =
  different company", and real aliases break that rule.
- **Real subsidiaries often list the parent's domain with a path.** `Microsoft Canada` lists
  `microsoft.com/en-ca`, and `Nintendo Hong Kong` lists `nintendo.com/hk/`. eTLD+1
  canonicalisation therefore gives them the same domain, the cannot-link constraint cannot fire,
  and whole subsidiary families merge. Of the engine's 1,231 false-merge pairs, 525 are sibling
  subsidiaries and 383 are parent/subsidiary pairs.

The synthetic set mostly modelled typos and suffix noise. It under-represented all three of these
failure modes, which is why its 0.974 does not transfer.

Process note: the first real run sampled only the subsidiary, rebrand and same-label groups
(3,718 records). It gave engine F1 0.364 against the baseline's 0.263. I then added the generic
slice, which was already in the plan ("businesses with an official website"), so the overall
number is not driven only by hard cases. Each slice is still reported above. Nothing in the
engine was changed in response to these results.

**Real error examples** (from `reports/real_eval.md`):

- *Miss (rebrand):* `Sony Music Canada` (sonymusic.ca) vs `CBS RECORDS CANADA LTD.`. Never
  blocked: the only shared word, `canada`, is not a blocking key for either name.
- *Miss (ticker alias):* `Bumi Resources` vs `BUMI`. Blocked, but p = 0.209.
- *Miss (extra words):* `Crown Resorts Limited` vs `Crown Limited`. p = 0.288, just under the
  threshold.
- *False merge (shared parent domain):* `Microsoft Sweden` (`news.microsoft.com/sv-se/`) with
  `Microsoft Canada` (`microsoft.com/en-ca`). p = 0.615 on a `domain_match`, and the merge then
  chains to `Microsoft Morocco`.
- *False merge (same name, different company):* `Capitalia` (capitalia.it) with `Capitalia SE`.
  p = 0.930. These are two different Wikidata entities with an identical brand and no domain on
  one side.

### Evaluation protocol (synthetic gold set)

- **Gold set:** [`data/gold/companies_gold.csv`](data/gold/companies_gold.csv) has
  **173 hand-labelled FICTIONAL records**, 70 entities and 156 true co-referent pairs. Each row
  carries a note naming its trap: subsidiaries with their own ccTLD (separate account), a rebrand
  shown only through `f/k/a`, `Inc` vs `Incorporated`, `Helio & Co` vs `Helio & Company Limited`,
  umlaut transliteration, shared `github.io`/`substack.com` hosts, ATS URLs used as the "domain",
  Google redirect URLs, an incubator domain listed by two portfolio startups, numbered fund series,
  and one generic name used by three different companies.
- **Training data is synthetic only** ([`synth.py`](src/signalgraph/resolution/synth.py), seed 13,
  6,891 records). It never sees the gold set.
- **The operating threshold is chosen on a separate synthetic validation set** (seed 99, 3,605
  records). The selected value, **0.40**, is then applied to the gold set unchanged. The gold set
  is not used for tuning.
- **Metrics are pairwise over all record pairs.** True pairs that blocking never proposed count as
  false negatives, so blocking recall is part of the reported recall.

### Results on the synthetic gold set (FICTIONAL records, measured by `make eval` → [`reports/er_eval.json`](reports/er_eval.json))

**v2 note.** The table below is the v1 engine (synthetic-only training, threshold 0.40). The v2
engine (threshold 0.10, model trained on synthetic + real DEV pairs) scores **P 0.939 / R 0.891 /
F1 0.915** (139 / 9 / 17) here, a regression of 0.059 F1; `reports/er_eval.json` now holds the v2
numbers. Fellegi-Sunter at 0.10 with the v2 features: P 0.942 / R 0.840 / F1 0.888.

| Method @ threshold 0.40 | Precision | Recall | **F1** | TP / FP / FN |
|---|---|---|---|---|
| Baseline: exact normalised name OR same domain | 0.800 | 0.769 | 0.784 | 120 / 30 / 36 |
| Fellegi-Sunter + constrained union-find | 0.930 | 0.942 | 0.936 | 147 / 11 / 9 |
| Logistic regression, pairwise decisions only | 0.897 | 0.897 | 0.897 | 140 / 16 / 16 |
| Logistic regression + plain transitive closure | 0.784 | 0.974 | 0.869 | 152 / 42 / 4 |
| **Logistic regression + constrained union-find (v1 shipped)** | **0.974** | **0.974** | **0.974** | **152 / 4 / 4** |

- **Blocking on the gold set:** 292 candidate pairs out of 14,878 (98.0% reduction), keeping
  96.8% of the true pairs. Transitive closure recovers most of the rest.
- **Best threshold:** a full sweep from 0.05 to 0.95 on the gold set peaks at F1 0.974 at 0.45,
  the same F1 as the threshold picked blind on synthetic data. The optimum is flat, so the
  validation-chosen threshold costs nothing here.
- **The cannot-link constraint matters more than the model choice.** It raises precision from
  0.784 to 0.974 at equal recall.

**The 8 remaining errors (worth reading):**

- 4 FP: `Ember & Oak Ventures Fund I, L.P.` merged with `Ember & Oak Ventures`. A fund vehicle
  and its manager share every token, and neither record has a domain.
- 2 FN: `NimbusGrid Energy` vs `Nimbus Grid Energy` (squashed tokens, no domain on one side).
- 1 FN: `Halcyon Tx` vs `Halcyon Biotherapeutics`. This would need a ticker or abbreviation
  dictionary.
- 1 FN: a startup whose only listed domain is its incubator's (`hatchery.vc/portfolio/…`). The
  cannot-link constraint correctly refuses the merge. Known limitation.

**Honesty notes:** two features, *residual similarity* and *number mismatch*, were added after
error analysis: the first came from a gold-set failure and the second from a live-run failure
(§7). Six numbered-series records were added to the gold set at the same time. The gold set
therefore influenced feature design, though not training or threshold choice. Synthetic-label
quality at scale is lower (F1 ≈ 0.90, see §6) because the generator injects heavier noise than
the gold set has.

## 5. Scoring model

```
points(signal) = base_weight[type] × subtype_weight × magnitude_factor × 0.5^(age_days / half_life[type])
score          = 100 × (1 − exp(−Σ capped_points_by_type / 40))
```

| Type | Half-life | Base | Subtype weights (excerpt) | Magnitude | Cap |
|---|---|---|---|---|---|
| funding (Form D) | 90 d | 25 | operating co 1.0, **pooled fund 0.0**, amendments ×0.5 | log10 USD (1M → 1×, 10M → 2×, max 3×) | – |
| hiring | 30 d | 3 | **gtm 2.0**, data 1.5, leadership 1.5, eng 1.0, ops 0.3 | – | 30 |
| news | 45 d | 6 | funding 2.0, leadership/expansion 1.5, launch 1.0 | – | – |
| eng_activity | 14 d | 1 | new public repo 3.0, release 2.0, push 0.5 | log1p(count) | 12 |

Everything lives in [`config/scoring.yaml`](config/scoring.yaml). ICP rules in
[`config/icp.yaml`](config/icp.yaml) (countries, excluded subtypes, name patterns such as
`Fund II`, minimum signals) **never change the score**. They set `icp_fit` and give a reason
for each failed rule.

Example from the offline demo (`GET /accounts/{id}/explain`, fictional company, scored as of
2026-11-19). The contributions always sum exactly to `raw_points`:

| Signal | Age | Base × subtype | Magnitude | Decay | Points |
|---|---|---|---|---|---|
| Form D, $12M sold | 7.0 d | 25 | 2.079 | 0.948 | 49.25 |
| News: "…raises $12M Series A…" | 6.4 d | 12 | 1.0 | 0.906 | 10.87 |
| News: "…appoints first Chief Revenue Officer" | 15.4 d | 9 | 1.0 | 0.789 | 7.10 |
| Hiring: Enterprise Account Executive (gtm) | 4.4 d | 6 | 1.0 | 0.903 | 5.42 |
| … 13 more | | | | | |
| **Northwind Robotics: raw 96.51 → score 91.04** | | | | | |

## 6. Quickstart

```bash
make install         # python3.11 venv + dev extras
make check           # ruff + mypy --strict + pytest
make real-eval       # ER on the REAL Wikidata gold set, held-out TEST split -> reports/real_eval_v2.{json,md}
make demo            # offline: replays recorded fixtures (FICTIONAL companies) end to end
make serve           # FastAPI on :8000 over the demo DB -> /docs
```

Or step by step:

```bash
signalgraph --db demo.duckdb ingest --fixtures fixtures --today 2026-11-19   # 35 signals
signalgraph --db demo.duckdb ingest --fixtures fixtures --today 2026-11-19   # inserted: 0
signalgraph --db demo.duckdb resolve                                         # 15 mentions -> 8 accounts
signalgraph --db demo.duckdb score --as-of 2026-11-19T00:00:00+00:00
signalgraph --db demo.duckdb export --format csv            # or clay | hubspot, --icp-only, --out
signalgraph --db live.duckdb ingest                         # live: uses config/watchlist.yaml
```

Demo output (real run, fictional data):

```
name                           domain                 score  icp_fit  n_signals
Northwind Robotics             northwindrobotics.com  91.04  True     17   # GH + GitHub + RSS + Form D under its legal name
Halcyon Biosciences            halcyonbio.com         80.13  True     4    # "HALCYON BIO, INC." (SEC, no domain) joined
Zentro                         zentro.ai              73.20  True     4    # Ashby + Lever board filed under old name + SEC f/k/a
BluePeak Health                bluepeakhealth.com     42.00  True     3    # "Blue Peak Health Inc" Form D/A (amendment x0.5)
Kestrel Data, Inc.             kestreldata.com        21.20  True     2    # kept apart from ...
Hyperion Vale Inc              hyperionvale.com       20.25  True     2    # domain came wrapped in a Google redirect URL
Kestrel Data Systems Pvt. Ltd. kestreldata.in         15.12  True     2    # ... the Indian company with the same name stem
Ember & Oak Ventures Fund I    -                       0.00  False    1    # pooled fund: weight 0, fails ICP name rule
```

API: `GET /health`, `GET /accounts?icp_only=&min_score=&limit=&offset=`, `GET /accounts/{id}`,
`GET /accounts/{id}/explain`, `GET /signals?signal_type=`. Configuration lives in
[`.env.example`](.env.example); no secrets are needed. `GITHUB_TOKEN` is optional.

**Exports.** `clay` writes flat JSON rows for a Clay webhook table. `hubspot` writes
`POST /crm/v3/objects/companies/batch/upsert` bodies in chunks of 100, keyed on a unique
custom property `signalgraph_account_id`, so re-exporting updates records instead of creating
duplicates.

## 7. Benchmarks (measured)

`make bench` runs the full production resolver end to end (normalise → block → score → cluster →
build accounts) on **synthetic** records. It does 3 repeats with caches cleared before each one
and reports the median. Machine: the 2-vCPU x86_64 Linux sandbox this was built in, Python
3.11.18, single process ([`reports/benchmark.json`](reports/benchmark.json)).

| Records | Median wall | **Records / s** | Candidate pairs (of all pairs) | Stage split (normalise / block / score / cluster) | Pairwise F1 vs synthetic labels |
|---|---|---|---|---|---|
| 10,000 | 2.35 s | **4,250** | 59,368 of 49,995,000 (0.12%) | 0.38 / 0.11 / 1.48 / 0.36 s | 0.907 |
| 50,000 | 12.38 s | **4,038** | 300,868 of 1,249,975,000 (0.024%) | 2.38 / 0.81 / 7.28 / 1.81 s | 0.896 |

Throughput stays almost flat from 10k to 50k records, which is what blocking is for: the number
of candidate pairs grows about linearly with the record count, not quadratically. Pair scoring
in pure Python accounts for about 60% of the time. Jaro-Winkler is memoised on ordered name pairs
(symmetry is property-tested), because the 10k run has only 26k distinct name pairs among its
59k candidates. Individual runs on this shared machine varied from 10.8 to 15.3 s at 50k.

## 8. Live data (what actually ran)

On **2026-11-19, about 07:52 UTC**, from this sandbox, I ran the live connectors.
[`reports/live_run_2026-11-19.json`](reports/live_run_2026-11-19.json) holds the counts; no raw
data is committed.

| What | Result |
|---|---|
| `pytest tests/live` (`SIGNALGRAPH_LIVE=1`) | 2 passed: EDGAR Form D and Greenhouse |
| Ingest #1 (EDGAR, 3-day lookback capped at 25 filings, plus Greenhouse board `stripe`) | **809 signals**: 784 Stripe postings and 25 Form D filings, in 19.9 s. One transient `ConnectError` on `www.sec.gov` was retried successfully |
| Ingest #2 (same config) | **0 inserted**. Both connectors started from their stored watermarks (8.8 s) |
| Role families across Stripe's 784 postings | gtm 267 · eng 195 · other 110 · ops 70 · product 56 · data 51 · security 20 · leadership 15 |
| Form D split | 17 operating companies, 8 pooled investment funds (tagged `form_d_fund`, weight 0). Median amount for operating companies: $1.6M |
| Scoring | 26 accounts, 17 ICP-fit; Stripe's hiring contribution hit its cap of 30 points |

**What the live run caught:** the first resolve turned 26 mentions into **20 accounts**. Every one
of those 6 merges was wrong. Blocking had produced only 18 candidate pairs, and all of them were
between numbered series vehicles from two sponsors (`<Sponsor> Partners 16/18/19 LLC`,
`<Sponsor> Ventures I…VI – Series …`). The fix was to add the `number_mismatch` feature, add
numbered-series hard negatives to the synthetic generator, and add 6 numbered-series records to
the gold set, then retrain and re-evaluate. Re-resolving the same 26 mentions gave **26
accounts**.

Not run live: GitHub (the sandbox egress proxy blocks `/orgs/{org}/events`), Lever, Ashby and RSS.
These four are covered by fixture tests only.

## 9. Limitations and trade-offs

- **Real-data accuracy is moderate (F1 0.592 on the held-out Wikidata TEST split, §4; v1 was
  0.393).** The main gaps are:
  - rebrands and ticker aliases with no shared token (they need an alias/ticker source),
  - chaining of generic names into a family at the low 0.10 threshold (`Italia` → `IBM Italia`),
  - the v2 model trusts a shared domain less, so same-domain rebrands without an alias can stay
    split (synthetic gold F1 fell from 0.974 to 0.915).
- **Domain cannot-link vs rebrands.** A rebrand whose old *and* new domains both appear stays
  split unless an `f/k/a` alias or a shared domain bridges it. I chose the precision win
  (+0.19 on the gold set) over that recall.
- **Fund vehicles vs managers** (`… Fund I, L.P.` vs `… Ventures`) still merge when neither side
  has a domain. The ICP name rule catches them for routing, but the graph is wrong.
- **Shared incubator domains**: a portfolio company that lists only `incubator.vc/portfolio/x` is
  treated as the incubator's domain.
- **Full re-resolve each run.** This is fine at 50k mentions (12 s), but it is not incremental
  (see roadmap). Account ids are keyed on domain, so they stay stable as clusters grow.
- **Scoring weights are expert priors**, not fitted to outcomes. There is no CRM outcome data in
  this project, and I chose not to invent any.
- **Form D lists no website**, so SEC-only accounts depend on name matching. The 3-day EDGAR
  lookback with a 25-filing cap is a politeness default, not a coverage claim.
- **Single writer.** DuckDB handles one batch writer; the API is read-only (one cursor per request).

## 10. Roadmap

1. Incremental ER: re-score only blocks touched by new mentions, and keep cluster history so
   splits and merges are auditable.
2. Active learning: queue the pairs closest to the threshold for human review and feed the
   labels back into training.
3. Learn the scoring weights from closed-won / closed-lost CRM outcomes once they exist, with the
   current weights as the baseline to beat.
4. Website enrichment for SEC-only accounts (domain discovery) to lift recall on Form D.
5. A Rust/`rapidfuzz` similarity backend behind the same interface; pair scoring is about 60% of
   ER time.
6. Scheduled runs (GitHub Actions cron) that post webhooks for newly ICP-fit accounts.

## Repository layout

```
src/signalgraph/
  domain/        models.py (SignalEvent, Account, ScoreBreakdown), taxonomy.py (role/news classifiers)
  connectors/    http.py (PoliteClient), edgar.py, jobs.py (Greenhouse/Lever/Ashby), github.py, rss.py, base.py
  resolution/    names.py, domains.py, features.py, blocking.py, scorers.py, cluster.py, engine.py,
                 synth.py, evaluate.py, er_model.json
  scoring/       config.py (typed YAML), engine.py
  storage/       schema.sql, repository.py (the only SQL)
  services/      pipeline.py (ingest/resolve/score), export.py (CSV/Clay/HubSpot)
  api/app.py     FastAPI        cli.py   click CLI
config/          scoring.yaml, icp.yaml, watchlist.yaml (live)
fixtures/        recorded API responses for FICTIONAL companies + manifest + demo watchlist
data/gold/       hand-labelled ER gold set (FICTIONAL)
data/real/       REAL gold set derived from Wikidata (CC0) + pull metadata; raw cache gitignored
benchmarks/real/ build_wikidata_gold.py (pull + build), make_split.py (DEV/TEST), tune_dev.py,
                 predict_split.py, run_split_eval.py (v2 eval), run_real_eval.py (v1 full-set eval)
scripts/         train_er.py, eval_er.py, bench_er.py
reports/         er_eval.json, real_eval_v2.{json,md}, real_tune_dev.json, real_eval.{json,md} (v1),
                 benchmark.json, live_run_2026-11-19.json
tests/           unit/ integration/ property/ (hypothesis) live/ (opt-in)
docs/adr/        0001 DuckDB · 0002 ER design · 0003 scoring model
```

## Data and honesty

- Every company in `fixtures/` and `data/gold/` is fictional. The synthetic generator is labelled
  as such everywhere it is used.
- `data/real/` holds real companies, derived from Wikidata (CC0 1.0, pulled 2026-11-20), with
  QIDs so every label can be checked.
- Every number in this README comes from a committed report or a run described above. There are
  no customer results, because there are no customers yet.

MIT licensed. Built by Hussain Singapuri.
