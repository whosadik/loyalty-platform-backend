# Roadmap Teacher Dataset Report

- planning_examples_total: **49421**
- stepwise_rows_total: **2239969**
- users_total: **12811**
- meaningful_non_trivial_length_share: **1.0**

## By Category
- fragrance: **12265**
- haircare: **12347**
- makeup: **12341**
- skincare: **12468**

## First Anchor Type
- blush: **1624**
- cleanser: **2002**
- cold_day: **2283**
- cold_evening: **3750**
- conditioner: **2471**
- essence: **1303**
- eye_cream: **1321**
- eyeshadow: **1300**
- foundation: **2424**
- hair_mask: **1940**
- hair_oil: **1644**
- leave_in: **2005**
- lipstick: **1527**
- mascara: **2390**
- mask: **1444**
- moisturizer: **1751**
- primer: **1642**
- scalp_serum: **1752**
- serum: **1897**
- setting_spray: **1434**
- shampoo: **2535**
- spf: **1412**
- toner: **1338**
- warm_day: **4318**
- warm_evening: **1914**

## Target Lengths
- 4: **15349**
- 5: **13150**
- 6: **20773**
- 7: **149**

## Fragrance Slots
- cold_day: **12265**
- cold_evening: **12265**
- warm_day: **12265**
- warm_evening: **12265**

## Splits
- counts: **{"test": 7464, "train": 34783, "val": 7174}**
- user_overlap_counts: **{"train_val": 0, "train_test": 0, "val_test": 0}**

## Edge Exclusions
- duplicate_later_purchase: **2064282**
- ga_user: **319516**

## Readiness
- fragrance_included_initial_planner: **yes** - Fragrance slot-level teacher examples are sufficient.
- haircare_only_initial_planner: **yes** - Haircare teacher anchors are large enough for an initial baseline.
- multi_category_initial_planner: **yes** - Teacher dataset covers multiple categories at usable scale.
