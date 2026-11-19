# ADR 0003: Additive, decayed, capped and explainable scoring (no black-box model)

- Status: accepted
- Date: 2026-11-19

## Context

Sales needs one number per account to sort by, and RevOps needs to answer
"why is this account 87?" without a data scientist. There are no labelled
outcomes (won/lost deals) in this project to train a propensity model on, and
inventing them would be dishonest.

## Decision

`points = base_weight[type] * subtype_weight * magnitude_factor * 0.5^(age / half_life[type])`

- Per-type half-lives (funding 90 d, hiring 30 d, news 45 d, GitHub 14 d) and
  `max_age_days` cut-offs, all in `config/scoring.yaml`.
- Magnitude: `log10` of USD for Form D amounts (1M -> 1x, 10M -> 2x, cap 3x),
  `log1p` for event counts.
- Per-type caps so one high-volume source (a 784-posting job board) cannot
  dominate.
- Final score `100 * (1 - exp(-raw / 40))`: monotone, bounded, and roughly
  linear for small accounts.
- ICP rules (`config/icp.yaml`) never change the score; they set
  `icp_fit` and list human-readable reasons.
- Every contribution (age, decay, weights, points) is persisted and served at
  `GET /accounts/{id}/explain`.

## Consequences

- Fully explainable and tunable in YAML; contributions sum exactly to the raw
  score (tested).
- Weights are expert priors, not fitted. Once CRM outcomes exist the same
  features can be fed to a calibrated model and these weights become the
  baseline to beat (roadmap).
