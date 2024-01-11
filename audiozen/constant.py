import math
from collections import namedtuple

import numpy as np
import torch


NEG_INF = torch.finfo(torch.float32).min
PI = math.pi
SOUND_SPEED = 343  # m/s
EPSILON = np.finfo(float).eps
MAX_INT16 = np.iinfo(np.int16).max

ArraySetup = namedtuple("ArraySetup", "arrayType, orV, mic_pos, mic_orV, mic_pattern")
dicit_array_setup = ArraySetup(
    arrayType="planar",
    orV=np.array([0.0, 1.0, 0.0]),
    mic_pos=np.array(
        (
            (0.96, 0.00, 0.00),
            (0.64, 0.00, 0.00),
            (0.32, 0.00, 0.00),
            (0.16, 0.00, 0.00),
            (0.08, 0.00, 0.00),
            (0.04, 0.00, 0.00),
            (0.00, 0.00, 0.00),
            (0.96, 0.00, 0.32),
            (-0.04, 0.00, 0.00),
            (-0.08, 0.00, 0.00),
            (-0.16, 0.00, 0.00),
            (-0.32, 0.00, 0.00),
            (-0.64, 0.00, 0.00),
            (-0.96, 0.00, 0.00),
            (-0.96, 0.00, 0.32),
        )
    ),
    mic_orV=np.tile(np.array([[0.0, 1.0, 0.0]]), (15, 1)),
    mic_pattern="omni",
)
line_dicit_array_setup = ArraySetup(
    arrayType="planar",
    orV=np.array([0.0, 1.0, 0.0]),
    mic_pos=np.array(
        (
            (-0.96, 0.00, 0.00),
            (-0.64, 0.00, 0.00),
            (-0.32, 0.00, 0.00),
            (-0.16, 0.00, 0.00),
            (-0.08, 0.00, 0.00),
            (-0.04, 0.00, 0.00),
            (0.00, 0.00, 0.00),
            (0.04, 0.00, 0.00),
            (0.08, 0.00, 0.00),
            (0.16, 0.00, 0.00),
            (0.32, 0.00, 0.00),
            (0.64, 0.00, 0.00),
            (0.96, 0.00, 0.00),
        )
    ),
    mic_orV=np.tile(np.array([[0.0, 1.0, 0.0]]), (13, 1)),
    mic_pattern="omni",
)

line_dicit_5mic_array_setup = ArraySetup(
    arrayType="planar",
    orV=np.array([0.0, 1.0, 0.0]),
    mic_pos=np.array(
        (
            (-0.08, 0.00, 0.00),
            (-0.04, 0.00, 0.00),
            (0.00, 0.00, 0.00),
            (0.04, 0.00, 0.00),
            (0.08, 0.00, 0.00),
        )
    ),
    mic_orV=np.tile(np.array([[0.0, 1.0, 0.0]]), (5, 1)),
    mic_pattern="omni",
)
