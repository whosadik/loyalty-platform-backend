# Roadmap Decision Surfaces

- raw_plan_refreshed_events: **7**
- excluded_noisy_decision_points_count: **0**
- excluded_legacy_bad_fragrance_completions_count: **0**

## Surface Types
- initial_refresh: raw=7, trusted=7, trusted_users=7, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0
- other_trusted_transition: raw=0, trusted=0, trusted_users=0, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0
- post_completed: raw=0, trusted=0, trusted_users=0, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0
- post_refresh_rebuild: raw=0, trusted=0, trusted_users=0, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0
- post_skipped: raw=0, trusted=0, trusted_users=0, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0

## Dataset Slices
- combined: trusted=7, positives=0, stop_rate=1.0, fragrance_positives=0
- continuation_only: trusted=0, positives=0, stop_rate=0.0, fragrance_positives=0
- initial_only: trusted=7, positives=0, stop_rate=1.0, fragrance_positives=0

## Readiness
- full_planner_with_continuation_transitions: **no** - Continuation positives are too sparse and category coverage is too weak.
- haircare_only_initial_planner: **no** - Too few trusted haircare initial positives for a stable baseline.
- multi_category_initial_planner: **no** - Initial positives are too sparse or too concentrated in one category.
