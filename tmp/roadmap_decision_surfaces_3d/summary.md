# Roadmap Decision Surfaces

- raw_plan_refreshed_events: **2333**
- excluded_noisy_decision_points_count: **2134**
- excluded_legacy_bad_fragrance_completions_count: **0**

## Surface Types
- initial_refresh: raw=22, trusted=13, trusted_users=13, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0
- other_trusted_transition: raw=0, trusted=0, trusted_users=0, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0
- post_completed: raw=9, trusted=9, trusted_users=3, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=8
- post_refresh_rebuild: raw=2311, trusted=186, trusted_users=88, non_stop_positive_share=0.048387, fragrance_share=0.0, step_advance_count=0
- post_skipped: raw=0, trusted=0, trusted_users=0, non_stop_positive_share=0.0, fragrance_share=0.0, step_advance_count=0

## Dataset Slices
- combined: trusted=208, positives=9, stop_rate=0.956731, fragrance_positives=0
- continuation_only: trusted=9, positives=0, stop_rate=1.0, fragrance_positives=0
- initial_only: trusted=199, positives=9, stop_rate=0.954774, fragrance_positives=0

## Readiness
- full_planner_with_continuation_transitions: **no** - Continuation positives are too sparse and category coverage is too weak.
- haircare_only_initial_planner: **no** - Too few trusted haircare initial positives for a stable baseline.
- multi_category_initial_planner: **no** - Initial positives are too sparse or too concentrated in one category.
