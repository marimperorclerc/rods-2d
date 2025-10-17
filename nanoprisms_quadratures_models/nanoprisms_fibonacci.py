r"""
This model describes the scattering of light by nanoprisms using the Fibonacci quadrature.

Definition
----------
Fibonacci quadrature is a numerical integration method that uses evaluation points spaced 
according to Fibonacci ratios to efficiently approximate an integral with minimal function evaluations.
This method is based on the generation of a set of points on the unit sphere and their associated weights.
Here, it is used to perform the orientational averaging of the scattering intensity. 
This method has been shown to be a good compromise between computational cost and accuracy.
The quadrature input parameter is the number of points to be generated on the sphere.

References
----------

Authorship and Verification
---------------------------

* **Authors:** Jules Marcone, Marianne Imperor-Clerc, Nicolas Ratel-Ramond, Mokhtari Sara **Date:** 07/09/2025
* **Last Modified by:** Sara Mokhtaru **Date:** 16/10/2025
* **Last Reviewed by:** **Date:**

"""
name = "nanoprisms_fibonacci"
title = "nanoprisms fibonnacci quadrature"
description = """
        nanoprisms fibonacci quadrature"""

category = "plugin"

import numpy as np
from numpy import inf
from prismformfactors import *
import numpy as np
import matplotlib.pyplot as plt
import time



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
              ["npoints_fibonacci",     "",           300, [1, 1e6],   "", 
                        "Number of points on the sphere for the Fibonacci integration"]
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


######################################## Fibonacci integration ################################################

def fibonacci_sphere(npoints_fibonacci: int):
    """
    Generates npoints quasi-uniformly distributed on the unit sphere
    in Cartesian coordinates (x,y,z) and their associated weights.
    Parameters
    ----------
    npoints : int
        Number of points to generate.
    Returns
    -------
    points : ndarray, shape (npoints, 3)
        Cartesian coordinates of the points on the unit sphere.
    weights : ndarray, shape (npoints,)
        Weights associated with each point for integration on the sphere.
    """

    indices = np.arange(0, npoints_fibonacci, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/npoints_fibonacci)        
    theta = 2 * np.pi * indices / ((1 + 5**0.5) / 2)  
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    weight = np.full(len(x),1/npoints_fibonacci)
    
    return np.column_stack((x, y, z)), weight


def plot_fibonacci_sphere(npoints_fibonacci=500, figsize=(7,7)):
    """
    3D representation of Fibonacci points on the unit sphere.
    Parameters
    ----------
    npoints : int
        Number of points to generate and display.
    figsize : tuple
        Size of the figure.

    """
    pts,w = fibonacci_sphere(npoints_fibonacci)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)

    # Axes et esthétique
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Fibonacci points on the unit sphere, ({npoints_fibonacci} total points)")
    ax.set_box_aspect([1,1,1])  # sphère non déformée
    plt.show()


def Iq(q, sld, sld_solvent, nsides:int, Rave, L, npoints_fibonacci:int=1000):
    """
    Integration over the unit sphere with Fibonacci points.
    Each point has an equal weight = 1/npoints.

    Parameters
    ----------
    q : float ou array
        Norm of the scattering vector
    sld, sld_solvent : 
        Contrast of scattering length density
    nsides, Rave, L : 
        Geometrical parameters of the prism
    npoints : int
        Number of Fibonacci points on the sphere
    Returns
    -------
    Iq : ndarray
        Scattering intensity averaged over all orientations
    time_fib : float
        Execution time of the function in seconds
    n_points_total : int
        Total number of points used in the quadrature
    """

    time_start = time.time()
    n_points_total = npoints_fibonacci
    nsides = int(nsides)
    q = np.atleast_1d(q)  # vecteur q

    
    q_unit,w = fibonacci_sphere(npoints_fibonacci)   # shape (npoints, 3)

    # Projections
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Compute intensity
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # Uniform average over the sphere
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)
    integral = np.mean(intensity, axis=1)
    time_end = time.time()
    time_fib = time_end - time_start
   
    print(f'Execution time Fibonacci with {npoints_fibonacci} points: {time_fib:.4f} seconds')

    return integral * (sld - sld_solvent)**2


Iq.vectorized = True




