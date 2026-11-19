# ADR 0002: Entity resolution = blocking + learned pair scorer + constrained union-find

- Status: accepted
- Date: 2026-11-19

## Context

The same company appears as `NORTHWIND ROBOTICS INC` (SEC), `Northwind
Robotics` (Greenhouse, with a domain), `northwind-robotics` (GitHub) and
`Northwind Robotcs` (a typo in a feed). Companies also rebrand, have country
subsidiaries with their own domains, share brand tokens with unrelated
companies, and file as numbered fund series. Comparing all n² pairs is not an
option beyond a few thousand records.

## Decision

1. **Canonicalise** names (unicode, legal forms, `f/k/a` aliases) and domains
   (eTLD+1 via the Public Suffix List *including* the private section;
   redirect unwrapping; free-mail / ATS / social hosts mapped to "unknown").
2. **Block** on three keys: exact canonical domain, the record's rarest name
   token (document frequency over the batch), and a 4-char name prefix.
   Oversized name blocks are skipped and counted, not silently truncated.
3. **Score** candidate pairs with an L2 logistic regression (IRLS) over nine
   features (Jaro-Winkler, TF-IDF cosine, containment, residual similarity,
   number mismatch, domain and country agreement/disagreement). Missing is
   neutral, disagreement is evidence. Fellegi-Sunter is implemented as the
   classical baseline and evaluated side by side.
4. **Train on synthetic data only.** The hand-labelled gold set is a held-out
   test set; the operating threshold is chosen on a *separate* synthetic
   validation set (seed 99), not on the gold set.
5. **Cluster** with union-find over edges sorted strongest-first, with a
   cannot-link constraint: never merge two clusters holding different
   canonical domains.

## Why the domain constraint

Measured on the gold set at the shipped threshold (`reports/er_eval.json`):
plain transitive closure scores P 0.78 / R 0.97, constrained closure scores
P 0.97 / R 0.97. Chaining through domain-less records (SEC filings) is the
dominant precision failure; the constraint removes most of it. The cost is
that a rebrand whose old and new domains both appear stays split unless an
alias or a shared domain bridges it.

## Consequences

- Model weights are a 1 KB JSON file under version control, readable in review.
- A feature change requires `make train eval` and a refreshed
  `reports/er_eval.json`; a test pins the shipped threshold to that report.
- Known failure classes (documented in the README) are incubator domains
  listed by portfolio companies, and parent/fund vehicles whose names contain
  the operating company's name.
