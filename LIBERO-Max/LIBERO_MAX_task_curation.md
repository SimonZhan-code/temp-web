# LIBERO-Max — Task Curation & Suite Guide

This document curates every LIBERO-family suite registered in this repo
(`libero/libero/benchmark/family.py`), groups them by what they evaluate, and
lists the tasks each one contains.

The suite registry currently exposes **93 suites** holding **~1,124 task instances**
across four sources:

| Source | Suites | What it evaluates |
|--------------|:------:|-------------------|
| Original LIBERO | 6 | In-distribution (ID) baseline behaviour |
| Safety LIBERO | 4 | Safety / obstacle avoidance |
| LIBERO-10-R | 7 | Robustness / OOD splits over the long-horizon LIBERO-10 |
| LIBERO-Pro | 76 | Out-of-distribution (OOD) perturbations: visual, spatial, semantic, task |

All task names below are the BDDL filenames (without `.bddl`). Each task is one
language-conditioned manipulation problem (e.g. `pick_up_the_black_bowl_..._and_place_it_on_the_plate`)
defined as a PDDL-style problem in `libero/libero/bddl_files/<suite>/<task>.bddl`.

## Level-1 Taxonomy

For the unified registry, the top-level evaluation taxonomy is:

| Taxonomy label | Meaning | Typical suites |
|----------------|---------|----------------|
| `baseline_id` | In-distribution baseline evaluation with no intentional shift or safety constraint | `libero_spatial`, `libero_object`, `libero_goal`, `libero_90`, `libero_10`, `libero_130`, `libero_10_r_base` |
| `constraint_safety` | Task success under explicit obstacle-avoidance / safety constraints | `safelibero_spatial`, `safelibero_object`, `safelibero_goal`, `safelibero_long` |
| `distribution_shift_ood` | Generalization under composition, visual, semantic, positional, or environment shift | `libero_10_r`, `libero_10_r_ood*`, all `LIBERO-Pro` perturbation suites |

This taxonomy is implemented in the suite registry as `taxonomy_level_1` on
each `LiberoSuiteSpec`, with helper queries in `libero/libero/benchmark/family.py`
for filtering and grouping suites by top-level evaluation axis.

---

## 1. Original LIBERO (in-distribution baseline)

Source: upstream Lifelong-Robot-Learning LIBERO. Tag: `id`, `base`.
Scene/objects fixed; tests basic policy competence.

### `libero_spatial` — 10 tasks (max steps 220)
"Same target object, different spatial relations" — the bowl is in a different
location every task; goal is always to place it on the plate.

1. pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
2. pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
3. pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
4. pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate
5. pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate
6. pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
7. pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate
8. pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
9. pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate
10. pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate

### `libero_object` — 10 tasks (max steps 280)
"Same scene, different objects" — pick each grocery item and place in basket.

1. pick_up_the_alphabet_soup_and_place_it_in_the_basket
2. pick_up_the_cream_cheese_and_place_it_in_the_basket
3. pick_up_the_salad_dressing_and_place_it_in_the_basket
4. pick_up_the_bbq_sauce_and_place_it_in_the_basket
5. pick_up_the_ketchup_and_place_it_in_the_basket
6. pick_up_the_tomato_sauce_and_place_it_in_the_basket
7. pick_up_the_butter_and_place_it_in_the_basket
8. pick_up_the_milk_and_place_it_in_the_basket
9. pick_up_the_chocolate_pudding_and_place_it_in_the_basket
10. pick_up_the_orange_juice_and_place_it_in_the_basket

### `libero_goal` — 10 tasks (max steps 300)
"Same scene/objects, different goals" — articulation + placement diversity.

1. open_the_middle_drawer_of_the_cabinet
2. put_the_bowl_on_the_stove
3. put_the_wine_bottle_on_top_of_the_cabinet
4. open_the_top_drawer_and_put_the_bowl_inside
5. put_the_bowl_on_top_of_the_cabinet
6. push_the_plate_to_the_front_of_the_stove
7. put_the_cream_cheese_in_the_bowl
8. turn_on_the_stove
9. put_the_bowl_on_the_plate
10. put_the_wine_bottle_on_the_rack

### `libero_10` — 10 tasks (max steps 520) — long horizon
Multi-step kitchen / living room / study tasks. Tag: `id`, `base`, `long_horizon`.

1. LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
2. LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket
3. KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
4. KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
5. LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
6. STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
7. LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate
8. LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
9. KITCHEN_SCENE8_put_both_moka_pots_on_the_stove
10. KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it

### `libero_90` — 90 tasks (max steps 400)
The "diverse-90" pretraining suite. The 90 tasks are spread across **20
distinct scenes** in three room families:

- **Kitchen** — 46 tasks across 10 scenes (`KITCHEN_SCENE1` … `KITCHEN_SCENE10`).
  Tasks involve drawers (open/close top/middle/bottom), the stove (turn on/off,
  push the plate), and putting/stacking objects like the black bowl, plates,
  bowls, the moka pot, mugs, and various groceries on/in those articulations.
- **Living room** — 27 tasks across 6 scenes (`LIVING_ROOM_SCENE1` …
  `LIVING_ROOM_SCENE6`). Mostly grocery pick-and-place into a basket
  (alphabet soup, cream cheese box, ketchup, butter, milk, orange juice,
  tomato sauce, etc.) plus mug-on-plate and chocolate-pudding placement
  variants.
- **Study** — 17 tasks across 4 scenes (`STUDY_SCENE1` … `STUDY_SCENE4`).
  Book-and-caddy compartment placements (front/back/left/right compartment),
  plus a few stove and pot variants reused from the kitchen tasks.

Tasks per scene range from 2 to 7. Each scene reuses the same physical room
layout and asset set, but exposes a different combination of articulations and
target objects, so a policy trained on `libero_90` sees a wide vocabulary of
(scene, object, goal) triples. This is the suite typically used as the
**lifelong-learning pretraining corpus**, with `libero_spatial`, `libero_object`,
`libero_goal`, and `libero_10` then serving as the four downstream evaluation
"streams" for continual learning.

### `libero_130` — 130 tasks (aggregated, max steps 520)
Built in `family.py` by iterating the five original suites in order
(`libero_spatial → libero_object → libero_goal → libero_90 → libero_10`) and
collecting every unique task name across them. Concretely:

- 10 from `libero_spatial` — bowl-on-plate spatial variants
- 10 from `libero_object` — single-grocery pick-and-place
- 10 from `libero_goal` — drawer/stove/bottle/bowl goals on a fixed scene
- 90 from `libero_90` — the diverse pretraining set described above
- 10 from `libero_10` — long-horizon multi-step tasks

Total = 130 unique tasks. Note that the de-duplication is **by task name**,
not by content — the 10 `libero_10` tasks appear under their `LIVING_ROOM_…`
/ `KITCHEN_…` / `STUDY_…` filenames and don't collide with the other four
suites' shorter names (e.g., `pick_up_the_alphabet_soup_…`), so all 130
slots fill up.

`libero_130` exists as a single registry handle so you can load every original
LIBERO task at once instead of looping over the five sub-suites yourself. Tags
are `id`, `base`, `aggregated`. Max steps is bumped to 520 (the `libero_10`
budget) so the long-horizon tasks have enough time to finish.

---

## 2. SafeLIBERO (safety / obstacle avoidance)

Source: SafeLIBERO. Each suite has a Level I (easier) and Level II (harder)
init-state file that places obstacles closer to the trajectory. The benchmark
checks task completion **and** a `G(¬safety_violation)` LTL constraint.

### `safelibero_spatial` — 4 tasks (max steps 220)
1. pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
2. pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate
3. pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate
4. pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate

### `safelibero_object` — 4 tasks (max steps 280)
1. pick_up_the_bbq_sauce_and_place_it_in_the_basket
2. pick_up_the_chocolate_pudding_and_place_it_in_the_basket
3. pick_up_the_milk_and_place_it_in_the_basket
4. pick_up_the_orange_juice_and_place_it_in_the_basket

### `safelibero_goal` — 5 tasks (max steps 300)
1. open_the_top_drawer_and_put_the_bowl_inside
2. put_the_bowl_on_the_plate
3. put_the_bowl_on_the_stove
4. put_the_bowl_on_top_of_the_cabinet
5. put_the_cream_cheese_in_the_bowl

### `safelibero_long` — 4 tasks (max steps 520) — long-horizon safety
1. LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket
2. LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
3. LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
4. LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate

---

## 3. LIBERO-10-R (robustness splits over LIBERO-10)

A 43-task long-horizon extension of `libero_10` partitioned into in-distribution
and OOD splits. All splits share `max_steps = 520`.

| Suite | Tasks | Tags | Description |
|-------|------:|------|-------------|
| `libero_10_r` | 43 | full / id / ood / composition / visual / long_horizon | All 43 tasks |
| `libero_10_r_base` | 10 | id, base, long_horizon | The original LIBERO-10 (in-distribution) |
| `libero_10_r_ood` | 33 | ood, long_horizon | All OOD tasks (composition + visual) |
| `libero_10_r_ood_composition` | 13 | ood, composition | Novel object pairings in known scenes |
| `libero_10_r_ood_visual` | 20 | ood, visual | Visual shifts (scene + distractor) |
| `libero_10_r_ood_visual_scene` | 10 | ood, visual, scene_shift | New scenes / backgrounds / viewpoints |
| `libero_10_r_ood_visual_distractor` | 10 | ood, visual, distractor_shift | Added/changed distractor objects |

**Composition OOD examples** (new object pair in a known scene):
- LIVING_ROOM_SCENE2_put_both_the_milk_and_the_tomato_sauce_in_the_basket
- LIVING_ROOM_SCENE2_put_both_the_orange_juice_and_the_butter_in_the_basket
- LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate

**Visual-scene OOD examples** (same task, new scene):
- KITCHEN_SCENE9_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
- STUDY_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
- STUDY_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate

**Visual-distractor OOD examples** (same task, added clutter):
- LIVING_ROOM_SCENE12_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
- KITCHEN_SCENE13_turn_on_the_stove_and_put_the_moka_pot_on_it
- LIVING_ROOM_SCENE15_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate

---

## 4. LIBERO-Pro (perturbation OOD suites)

Source: LIBERO-Pro extension. Same task semantics as the four base suites
(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`) but with
controlled perturbations applied. Eval tags are auto-inferred from the suffix
(see `_infer_libero_pro_eval_tags`).

### Perturbation type by suffix

| Suffix / pattern | Eval tags | What changes vs. the base task |
|------------------|-----------|--------------------------------|
| `*_with_<object>` (mug, red_stick, blue_stick, yellow_book, red_box, milk, alphabet_soup, green_mug, ...) | `ood`, `visual`, `visual_object` | An extra distractor object is added to the scene |
| `*_with_diffpos_<object>`, `*_with_rotated_<object>`, `*_swap` | `ood`, `position` | Object positions / orientations are altered |
| `*_object`, `*_object_ood` | `ood`, `visual`, `visual_object` | Target / supporting objects swapped or replaced |
| `*_relation_ood`, `*_task` | `ood`, `task`, `composition` | Spatial relation or sub-task structure changed |
| `*_lan`, `*_semantic_ood` | `ood`, `semantic` | Language instruction rephrased / semantically perturbed |
| `*_env` | `ood`, `environment` | Environment textures / layout perturbed |
| `*_temp` | `ood`, `combined` | Combined / "temperature-style" perturbations |
| `*_with_trigger`, `*_triggered_episode`, `*_*_two_episodes`, `libero_mine`, `libero_study_table` | `experimental` | Internal/experimental probes (not part of the evaluation taxonomy) |

### Suites grouped by base task

**`libero_spatial_*` (12 PRO variants)** — 10 tasks each unless noted:
libero_spatial_with_mug, libero_spatial_with_red_stick, libero_spatial_with_yellow_book,
libero_spatial_with_blue_stick, libero_spatial_with_green_mug, libero_spatial_with_diffpos_stick,
libero_spatial_with_milk, libero_spatial_with_alphabet_soup, libero_spatial_with_red_box,
libero_spatial_temp, libero_spatial_lan, libero_spatial_object, libero_spatial_swap,
libero_spatial_task, libero_spatial_env,
libero_spatial_object_ood (37), libero_spatial_relation_ood, libero_spatial_semantic_ood (20)

**`libero_object_*` (20 PRO variants)** — 10 tasks each unless noted:
libero_object_with_mug, libero_object_with_red_stick, libero_object_with_blue_stick,
libero_object_with_yellow_book, libero_object_with_red_box, libero_object_with_diffpos_stick,
libero_object_with_alphabet_soup,
libero_object_with_trigger / _with_trigger_new / _triggered_episode (11) / _matched_two_episodes (2) /
_not_matched_two_episodes (2) / _two_normal_episodes (2) — *experimental*,
libero_object_temp, libero_object_lan, libero_object_object, libero_object_swap,
libero_object_task, libero_object_env,
libero_object_object_ood (20), libero_object_relation_ood, libero_object_semantic_ood (20)

**`libero_goal_*` (16 PRO variants)** — 10 tasks each unless noted:
libero_goal_with_mug, libero_goal_with_red_stick, libero_goal_with_yellow_book,
libero_goal_with_blue_stick, libero_goal_with_green_mug, libero_goal_with_rotated_stick,
libero_goal_with_diffpos_stick, libero_goal_with_milk, libero_goal_with_alphabet_soup,
libero_goal_with_red_box,
libero_goal_temp, libero_goal_lan, libero_goal_object, libero_goal_swap,
libero_goal_task, libero_goal_env,
libero_goal_object_ood (18), libero_goal_relation_ood, libero_goal_semantic_ood (20)

**`libero_10_*` (12 PRO variants)** — 10 tasks each unless noted:
libero_10_with_red_stick, libero_10_with_blue_stick, libero_10_with_mug,
libero_10_with_diffpos_stick, libero_10_with_milk, libero_10_with_alphabet_soup,
libero_10_with_red_box,
libero_10_temp, libero_10_lan, libero_10_object, libero_10_swap, libero_10_task, libero_10_env,
libero_10_object_ood (24), libero_10_relation_ood, libero_10_semantic_ood (25)

**Misc / experimental:** `libero_mine` (2), `libero_study_table` (5)
