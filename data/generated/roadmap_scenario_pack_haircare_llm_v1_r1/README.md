LLM-derived roadmap scenario pack: haircare_llm_v1

Source JSON: C:\Users\Adik\Uilesim\loyalty-platform-backend\data\roadmap_haircare_llm_v1.json

Validate:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_llm_v1_r1 --dry-run

Import into disposable DB:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_llm_v1_r1 --truncate --i-understand-its-destructive

Expected next distribution: {'conditioner': 6, 'hair_mask': 6, 'hair_oil': 1, 'leave_in': 10, 'scalp_serum': 7}
Outcome tags: {'clicked_no_purchase': 4, 'completed_exact': 11, 'completed_semantic': 9, 'exposed_no_click': 3, 'skipped': 3}