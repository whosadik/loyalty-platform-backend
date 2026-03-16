LLM-derived roadmap scenario pack: haircare_llm_v2_hardcases

Source JSON: C:\Users\Adik\Uilesim\loyalty-platform-backend\data\roadmap_haircare_llm_v2_hardcases.json

Validate:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_llm_v2_hardcases_r1 --dry-run

Import into disposable DB:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_llm_v2_hardcases_r1 --truncate --i-understand-its-destructive

Expected next distribution: {'conditioner': 4, 'hair_mask': 4, 'hair_oil': 2, 'leave_in': 10, 'scalp_serum': 4}
Outcome tags: {'clicked_no_purchase': 3, 'completed_exact': 6, 'completed_semantic': 9, 'exposed_no_click': 3, 'skipped': 3}