Synthetic roadmap scenario pack: haircare_hardcases_v2

Validate:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_hardcases_v2_r4 --dry-run

Import into disposable DB:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_hardcases_v2_r4 --truncate --i-understand-its-destructive

Expected next distribution: {'conditioner': 4, 'hair_oil': 4, 'leave_in': 8, 'scalp_serum': 4}
Outcome tags: {'completed_exact': 16, 'completed_semantic': 4}