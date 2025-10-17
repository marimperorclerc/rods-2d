r"""
This model describes the scattering of light by nanoprisms using the lebedev quadrature.

Definition
----------
The Lebedev quadrature is an approximation to the surface integral of a function over a three-dimensional sphere.
The grid is constructed so to have octahedral rotation and inversion symmetry. 
The number and location of the grid points together with a corresponding set of integration 
weights are determined by enforcing the exact integration of polynomials (or equivalently, spherical harmonics) up to a given order, 
leading to a sequence of increasingly dense grids analogous to the one-dimensional Gauss–Legendre scheme.
This method is based on the generation of a set of points on the unit sphere and their associated weights.
Here, it is used to perform the orientational averaging of the scattering intensity. 
The quadrature input parameter is the number of index of the order of the Lebedev quadrature (see orderlist).

References
----------

Authorship and Verification
---------------------------

* **Authors:** Jules Marcone, Marianne Imperor-Clerc, Nicolas Ratel-Ramond, Mokhtari Sara **Date:** 07/09/2025
* **Last Modified by:** Sara Mokhtaru **Date:** 16/10/2025
* **Last Reviewed by:** **Date:**

"""
name = "nanoprisms_lebedev"
title = "nanoprisms lebedev quadrature"
description = """
        nanoprisms lebedev quadrature"""

category = "plugin"

import numpy as np
from numpy import inf
from pylebedev import PyLebedev
from prismformfactors import *
import numpy as np
import matplotlib.pyplot as plt
import time

# List of the different quadrature orders in Lebedev
orderlist=[3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131]

leblib = PyLebedev()

#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [["sld", "1e-6/Ang^2", 126., [-inf, inf], "sld",
               "nanoprism scattering length density"],
              ["sld_solvent", "1e-6/Ang^2", 9.4, [-inf, inf], "sld",
               "Solvent scattering length density"],
              ["n_sides", "", 5, [3, 50], "",
               "n_sides"],
              ["R_ave", "Ang", 500, [0., inf], "volume",
               "Average radius"],
              ["L", "Ang", 5000, [0., inf], "volume",
               "length"],
              ["norder_index",     "",        20, [0, 31],   "",       
                  "Order index for the Lebedev quadrature, see orderlist"]
               ]


def volume(nsides,Rave,L):
    nsides=int(nsides)
    edge = functions.edge_from_gyration_radius(nsides,Rave)
    return functions.volume_nanoprism(nsides,edge,L)

def Iqabc(qa,qb,qc,nsides,Rave,L): # proportionnal to the volume**2
    nsides=int(nsides)
    edge = functions.edge_from_gyration_radius(nsides,Rave)
    intensity=formfactor.I_nanoprism([qa,qb,qc],nsides,edge,L)
    return intensity



######################################## Lebedev integration ##################################################

def lebedev_sphere(norder_index: int):
    """
    Generates quasi-uniformly distributed points on the unit sphere
    in Cartesian coordinates (x,y,z) and their associated weights.
    Parameters
    ----------
    norder : int
        index of the order of the Lebedev quadrature (see orderlist).
    Returns
    -------
    points : ndarray, shape (npoints, 3)
        Cartesian coordinates of the points on the unit sphere.
    weights : ndarray, shape (npoints,)
        Weights associated with each point for integration on the sphere.

    """

    order = orderlist[int(norder_index)]
    q_unit, w = leblib.get_points_and_weights(order)  # shape (npoints,3)

    return q_unit, w


def plot_lebedev_sphere(norder_index, figsize=(7,7)):
    """
    3D representation of Lebedev points on the unit sphere.
    Parameters
    ----------
    norder : int
        Index in orderlist for the Lebedev order.
    figsize : tuple
        Size of the figure.
    """
    pts, w = lebedev_sphere(norder_index)  # shape (npoints,3)
    order = orderlist[int(norder_index)]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)

    # Axes et esthétique
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Lebedev points on the unit sphere  (order of the polynom: {order}, {len(pts)} total points)")
    ax.set_box_aspect([1,1,1])  # sphère non déformée
    plt.show()



def Iq(q, sld, sld_solvent, nsides:int, Rave, L, norder_index:int):
    """
    Orientation integration on the unit sphere with Lebedev points.
    Parameters
    ----------
    q : float ou array
        Scattering vector
    sld, sld_solvent : contrast of scattering length density
    nsides, Rave, L : geometrical parameters of the prism
    norder : int
        Index in orderlist for the Lebedev order
    Returns
    -------
    Iq : ndarray
        Scattering intensity averaged over all orientations
    time_lebedev : float
        Execution time of the function in seconds
    n_points_total : int
        Total number of points on the sphere
    """

    time_start = time.time()
    nsides = int(nsides)

    q_unit, w = lebedev_sphere(norder_index)  # shape (npoints,3)
    n_points_total = len(q_unit)
    q = np.atleast_1d(q)  # make sure q is a 1D table

    # qa, qb, qc have the shape (nq, npoints)
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Compute intensity: Iqabc need to have qa, qb and qc with the same shape
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # sum over the points of Lebedev quadrature (axis=1) for each q
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)
    time_end = time.time()
    time_lebedev = time_end - time_start

    print(f'Execution time Lebedev with {len(q_unit)} points: {time_lebedev:.4f} seconds')

    return integral * (sld - sld_solvent)**2

Iq.vectorized = True  # Iq can be vectorized



