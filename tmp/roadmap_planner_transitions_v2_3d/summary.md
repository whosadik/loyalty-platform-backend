# Roadmap Planner Transitions Dataset Summary

- rows: **1475**
- decision points: **208**
- episodes: **199**
- users: **97**
- stop rate: **0.956731**
- excluded noisy decision points: **2134**
- excluded legacy bad fragrance completions: **0**

## Decision Types
- initial_refresh: **13**
- post_completed: **9**
- post_refresh_rebuild: **186**

## Positives By Label Source
- roadmap_completed_event: **1**
- roadmap_completed_exact: **8**
- stop_no_progress: **198**
- terminal_after_outcome_stop: **1**

## Slices
- combined: trusted=208, positives=9, stop_rate=0.956731, fragrance_positives=0
- continuation_only: trusted=9, positives=0, stop_rate=1.0, fragrance_positives=0
- initial_only: trusted=199, positives=9, stop_rate=0.954774, fragrance_positives=0

## Excluded Counts
- no_current_next_initial: **2133**
- no_snapshot_after_refresh: **1**

## Readiness
- full_planner_with_continuation_transitions: **no** - Continuation positives are too sparse and category coverage is too weak.
- haircare_only_initial_planner: **no** - Too few trusted haircare initial positives for a stable baseline.
- multi_category_initial_planner: **no** - Initial positives are too sparse or too concentrated in one category.
