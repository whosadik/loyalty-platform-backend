Synthetic roadmap scenario pack: haircare_v1

Validate:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_v1_r4 --dry-run

Import into disposable DB:
python manage.py import_synth_dataset --path C:\Users\Adik\Uilesim\loyalty-platform-backend\data\generated\roadmap_scenario_pack_haircare_v1_r4 --truncate --i-understand-its-destructive

Expected next distribution: {'conditioner': 8, 'hair_mask': 8, 'leave_in': 4, 'scalp_serum': 4}
Outcome tags: {'completed_exact': 16, 'completed_semantic': 4, 'no_conversion': 4}