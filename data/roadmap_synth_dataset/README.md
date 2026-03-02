Synthetic dataset for Üilesim (Roadmap/Offers/Recs) — generated 2026-03-02T10:00:00Z

Time window:
- Most activity occurs in: 2025-12-02T10:00:00Z .. 2026-03-02T10:00:00Z (UTC)

Files:
- users.csv: user_id, username, segment, favorite_category
- customer_profiles.csv: per-user profile JSON columns (goals/avoid_flags/hair_profile/makeup_profile/fragrance_profile)
- products.csv: Product-like rows (category/product_type + attrs/raw_meta JSON)
- transactions.csv + transaction_items.csv: purchase history
- owned_products.csv: aggregated ownership from transaction_items
- roadmap_plans.csv + roadmap_steps.csv: active plans and steps (fragrance steps use slots warm_day/warm_evening/cold_day/cold_evening)
- campaign_budgets.csv + offers.csv: offer definitions
- offer_assignments.csv + offer_events.csv: assignments and events (includes roadmap_shortcut targets; fragrance slot targets include actual_product_type)
- roadmap_events.csv: step_exposed/clicked/skipped (mostly sourced from offers exposures)
- recommendation_events.csv: home feed events

Import order (recommended):
1) products.csv
2) users.csv + customer_profiles.csv
3) transactions.csv + transaction_items.csv
4) owned_products.csv (or derive inside DB)
5) roadmap_plans.csv + roadmap_steps.csv
6) campaign_budgets.csv + offers.csv
7) offer_assignments.csv + offer_events.csv
8) roadmap_events.csv
9) recommendation_events.csv
