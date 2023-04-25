import random
from dataclasses import dataclass

import gpuRIR
import numpy as np
from numpy.typing import NDArray

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)


@dataclass
class Room:
    size: NDArray
    abs_weights: NDArray
    rt60: float
    att_ism: float = 15.0  # Attenuation when start using diffuse model [dB]
    att_max: float = 60.0  # Attenuation when start using max model [dB]


class RandomScalar:
    def __init__(self, para_range: list[float]) -> None:
        assert len(para_range) == 2, "Min and max"
        self.min, self.max = para_range

    def sample(self) -> float:
        """Generate a random value within the given range.
        Examples:
            >>> RandomParameter([[2, 3, 5] ,[4, 5, 6]]).get_value()
            >>> [3, 5, 6]
        """
        return self.min + ((self.max - self.min) * random.random())


class Random2DScalar:
    def __init__(self, para_range: list[list[float]]):
        assert len(para_range) == 2, "The parameter range must be a 2-D list."
        self.min = np.array(para_range[0], dtype=np.float32)
        self.max = np.array(para_range[1], dtype=np.float32)

    def sample(self) -> NDArray:
        return self.min + ((self.max - self.min) * np.random.random(self.min.shape))


class RoomLoader:
    """Generate room class using the given room parameters."""

    def __init__(self, conf) -> None:
        self.room_size_range = Random2DScalar(conf.room_size_range)
        self.room_absorption_coefficient = Random2DScalar(
            conf.room_absorption_coefficient
        )
        self.t60_list = conf.t60_list

    def sample(self) -> Room:
        return Room(
            size=self.room_size_range.sample(),
            abs_weights=self.room_absorption_coefficient.sample(),
            rt60=random.choice(self.t60_list),
        )


class Region2D:
    def __init__(self, bottom_left_coord, top_right_coord):
        """
        Construct a region using given parameters, i.e., bottom_left_coord and top_right_coord

        Notes:
            Top Left            Top Right
                +------+-----+------+
                |  1-1 | 1-2 | 1-3  |
                +------+-----+------+
                |  2-1 | 2-2 | 2-3  | (Depth)
                +------+-----+------+
                |  3-1 | 3-2 | 3-3  |
                +------+-----+------+
            bottom left       Bottom Right
                      (Width)

        Args:
            bottom_left_coord: array_like (x, y)
            top_right_coord: array_like, (x, y)
        """
        super(Region2D, self).__init__()
        # Infer all coordinates using bottom_left_coord and top_right_coord
        self.bottom_left_corner = np.array([bottom_left_coord[0], bottom_left_coord[1]])
        self.bottom_right_corner = np.array([top_right_coord[0], bottom_left_coord[1]])
        self.top_right_corner = np.array([top_right_coord[0], top_right_coord[1]])
        self.top_left_corner = np.array(
            [bottom_left_coord[0], self.top_right_corner[1]]
        )

        # Width and depth in civil engineering
        # (开间)    (进深)
        self.width = self.bottom_right_corner[0] - self.bottom_left_corner[0]
        self.depth = self.top_left_corner[1] - self.bottom_left_corner[1]

    def get_random_point(self, dist_from_wall: float = 0.5):
        """Get the coordinate of a given point within the region."""
        x = np.random.uniform(
            self.bottom_left_corner[0] + dist_from_wall,
            self.bottom_right_corner[0] - dist_from_wall,
        )
        y = np.random.uniform(
            self.bottom_left_corner[1] + dist_from_wall,
            self.top_left_corner[1] - dist_from_wall,
        )
        return np.array([x, y])

    def get_center_point(self):
        x = np.mean([self.bottom_left_corner[0], self.bottom_right_corner[0]])
        y = np.mean([self.bottom_left_corner[1], self.top_left_corner[1]])
        return np.array([x, y])

    def get_control_coords(self, num_control_points):
        """Separate all axes equally using a given num_control_points

        Examples
            return: np.array([
                [2, 4],
                [3, 5],
                x, y
            ])
        """
        x = np.linspace(
            self.bottom_left_corner[0],
            self.bottom_right_corner[0],
            num=num_control_points + 2,
        )[1:-1]
        y = np.linspace(
            self.bottom_left_corner[1],
            self.top_left_corner[1],
            num=num_control_points + 2,
        )[1:-1]

        return np.array([x, y]).transpose()  # [num_control_points, 2]

    def __repr__(self):
        return f"width: {self.width:.1f}, depth: {self.depth:.1f}"


class StaticGenerator:
    def __init__(self, region: NDArray, speaker_height: float) -> None:
        super().__init__(region)
        self.speaker_height = speaker_height
        self.region = Region2D(region[0], region[1])

    @staticmethod
    def distance2duration(distance: float, moving_speed: float) -> float:
        """Convert a distance to the fpath based on a giving moving speed.
        Args:
            distance: the unit is meter
            moving_speed: the unit is meter per second
        Returns:
            The fpath for a given distance. The unit is second.
        """
        return distance / moving_speed

    @staticmethod
    def euclidean_distance(x, y) -> float:
        return np.sqrt(np.sum(np.power(x - y, 2)))

    def generate(self):
        static_position = self.region.get_random_point()  # [x, y]
        return (static_position[None, ...], static_position[None, ...], {})


def generate_rir():
    room_size = np.array([10, 10, 10])


def main():
    sr = 16000
    speaker_height_range = RandomScalar([1.3, 2])
    rt60_range = RandomScalar([0.2, 0.8])
    room_size_range = Random2DScalar([[3, 3, 3.05], [12, 12, 4.05]])
    room_absorption_coefficient = Random2DScalar(
        [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    t60_range = Random2DScalar([0.2, 0.8])
    num_sources = 5
    distance_from_wall = 0.5

    for source_idx in range(num_sources):
        speaker_height = speaker_height_range.sample()
        room = Room(
            size=room_size_range.sample(),
            abs_weights=room_absorption_coefficient.sample(),
            rt60=rt60_range.sample(),
        )

    rir = generate_rir(
        room_size=room_size_range.sample(),
    )
