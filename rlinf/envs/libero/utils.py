# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
from typing import Union

import numpy as np
from libero.env_wrapper.utils import get_benchmark_overridden  # noqa: F401  # re-export


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extracts image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment

    Returns:
        Preprocessed image as numpy array
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(
    obs: dict[str, np.ndarray], resize_size: Union[int, tuple[int, int]] = 224
) -> np.ndarray:
    """
    Extracts wrist camera image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment
        resize_size: Target size for resizing

    Returns:
        Preprocessed wrist camera image as numpy array
    """
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
