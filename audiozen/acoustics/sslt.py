"""This module contains functions about Sound source localization and tracking (SSLT)"""
import torch


def cart2sph(cart, include_r=False):
    """Convert tensor-like cartesian coordinates to spherical coordinates.

    - cartesian: [x, y, z] => spherical: [elevation, azimuth, radius]
    - cartesian: [x, y] => spherical: [azimuth, radius]

    Args:
        cart: cartesian_coordinates
        include_r: the radius is optional

    Shapes:
        cart: [B, cartesian_coordinates, T], where cartesian_coordinates can be 2 (x,y) or 3 (x, y, z).
        sph: [B, spherical_coordinates, T], where spherical_coordinates can be 1 or 2.

    Returns:
        dim of spherical_coordinates can be 1 and 2.

    Note:
        Corresponding order relationship of cartesian and spherical coordinates are as follows:
        - 3D array: [x, y , z] <=> [elevation, azimuth, radius] <=> [theta, phi, r]
        - 2D array: [x, y] <=> [azimuth, radius] <=> [phi, r]
    """
    _, num_coordinates, _ = cart.shape  # [B, 2 or 3, T]
    assert (
        num_coordinates == 2 or num_coordinates == 3
    ), "Only support 2D or 3D coordinates."

    radius = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=1))  # [B, 1, T]
    phi = torch.atan2(cart[:, 1, :], cart[:, 0, :])  # [B, 1, T]

    if num_coordinates == 2:
        sph = torch.stack((phi, radius), dim=-1) if include_r else phi
    else:
        theta = torch.acos(cart[:, 2, :] / radius)
        sph = (
            torch.stack((theta, phi, radius), dim=-1)
            if include_r
            else torch.stack((theta, phi), dim=-1)
        )

    return sph


def sph2cart(sph):
    """Tensor-like spherical coordinates to cartesian coordinates.

        1. sphere: [elevation, azimuth] => cartesian: [x, y, z]
        2. sphere: azimuth => cartesian: [x, y]

     Shapes:
        sph: [B, spherical_coordinates, T], where spherical_coordinates can be 1 or 2.
        cart: [B, cartesian_coordinates, T], where cartesian_coordinates can be 2 (x, y) or 3 (x, y, z).

    Returns:
         dim of spherical_coordinates can be 1, 2 or 3.

    Notes:
        Corresponding order relationship of cartesian and spherical coordinates are as follows:

            - 3D array: (x, y, z) <=> (elevation, azimuth, radius) <=> (theta, phi, r)
            - 2D array: (x, y) <=> (azimuth, radius) <=> (phi, r)

        The output is the **unity cartesian**, i.e., the radius is supposed to be 1.
    """
    _, num_coordinates, _ = sph.shape  # [B, 1 or 2, T]
    assert (
        num_coordinates == 1 or num_coordinates == 2
    ), "Only support 1D or 2D coordinates now."

    if num_coordinates == 1:
        x = torch.cos(sph[:, 0, :])
        y = torch.sin(sph[:, 0, :])
        return torch.stack((x, y), dim=1)
    else:
        x = torch.sin(sph[:, 0, :]) * torch.cos(sph[:, 1, :])
        y = torch.sin(sph[:, 0, :]) * torch.sin(sph[:, 1, :])
        z = torch.cos(sph[:, 0, :])
        return torch.stack((x, y, z), dim=1)
