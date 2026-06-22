"""Unified LIBERO-family task maps.

This module keeps the evaluation taxonomy in one place:
- original LIBERO suites come from the upstream LIBERO task map
- SafeLIBERO suites come from the safety benchmark task map
- LIBERO-Pro suites are included via the main libero_suite_task_map
- LIBERO-10-R is defined here as explicit splits over the 43-task extension

The LIBERO-10-R split boundaries follow the upstream NVlabs ordering:
- first 10 tasks: original LIBERO-10
- next 13 tasks: novel composition OOD tasks
- next 10 tasks: visual OOD scene/background/viewpoint shifts
- final 10 tasks: visual OOD distractor/object shifts
"""

from __future__ import annotations

try:
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.libero.benchmark.safelibero_suite_task_map import safelibero_task_map
except ImportError:
    from libero.benchmark.libero_suite_task_map import libero_task_map
    from libero.benchmark.safelibero_suite_task_map import safelibero_task_map


ORIGINAL_LIBERO_TASK_MAP = libero_task_map
SAFELIBERO_TASK_MAP = safelibero_task_map


ORIGINAL_LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
)

SAFELIBERO_SUITES = (
    "safelibero_spatial",
    "safelibero_object",
    "safelibero_goal",
    "safelibero_long",
)


LIBERO_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_90": 400,
    "libero_10": 520,
    "libero_130": 520,
    "safelibero_spatial": 220,
    "safelibero_object": 280,
    "safelibero_goal": 300,
    "safelibero_long": 520,
    "libero_10_r": 520,
    "libero_10_r_all": 520,
    "libero_10_r_base": 520,
    "libero_10_r_ood": 520,
    "libero_10_r_ood_composition": 520,
    "libero_10_r_ood_visual": 520,
    "libero_10_r_ood_visual_scene": 520,
    "libero_10_r_ood_visual_distractor": 520,
}


LIBERO_10_R_ALL = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_milk_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_orange_juice_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_orange_juice_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_tomato_sauce_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_milk_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE1_put_both_the_ketchup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE1_put_both_the_tomato_sauce_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_left_of_the_plate",
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "KITCHEN_SCENE9_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "KITCHEN_SCENE9_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "STUDY_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "LIVING_ROOM_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "STUDY_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "KITCHEN_SCENE10_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "KITCHEN_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE2_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "STUDY_SCENE8_put_both_moka_pots_on_the_stove",
    "STUDY_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "LIVING_ROOM_SCENE12_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE12_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE13_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE14_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE15_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE11_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE16_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE11_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE18_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE16_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)

LIBERO_10_R_BASE = tuple(ORIGINAL_LIBERO_TASK_MAP["libero_10"])
LIBERO_10_R_OOD_COMPOSITION = LIBERO_10_R_ALL[10:23]
LIBERO_10_R_OOD_VISUAL_SCENE = LIBERO_10_R_ALL[23:33]
LIBERO_10_R_OOD_VISUAL_DISTRACTOR = LIBERO_10_R_ALL[33:43]
LIBERO_10_R_OOD_VISUAL = (
    *LIBERO_10_R_OOD_VISUAL_SCENE,
    *LIBERO_10_R_OOD_VISUAL_DISTRACTOR,
)
LIBERO_10_R_OOD = (
    *LIBERO_10_R_OOD_COMPOSITION,
    *LIBERO_10_R_OOD_VISUAL,
)
