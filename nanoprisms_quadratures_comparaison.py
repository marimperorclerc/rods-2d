r"""
This model describes the scattering of light by nanoprisms using a combination of numerical integration techniques.

Definition
----------
Three numerical integration methods are implemented to perform the orientational averaging
of the scattering intensity of a nanoprism over the unit sphere:
- Fibonacci quadrature, based on a Fibonacci lattice
- Lebedev quadrature, based on the Lebedev grid
- Gauss-Legendre quadrature, based on Gauss-Legendre nodes
These methods are based on the generation of a set of points on the unit sphere and their associated weights.
The inputs parameters are the geometrical parameters of the nanoprisms (number of sides, average radius and length) and
the parameters of the quadrature (number of total points on the sphere for Fibonacci, number of points for each angle phi and theta
for Gauss-Legendre, index of the order for Lebedev).

References
----------

Authorship and Verification
---------------------------

* **Authors:** Jules Marcone, Marianne Imperor-Clerc, Nicolas Ratel-Ramond, Mokhtari Sara **Date:** 07/09/2025
* **Last Modified by:** Sara Mokhtaru **Date:** 16/10/2025
* **Last Reviewed by:** **Date:**

"""
name = "nanoprisms"
title = "nanoprisms"
description = """
        nanoprisms"""

category = "plugin"

import numpy as np
from numpy import inf
from pylebedev import PyLebedev
from prismformfactors import *
import numpy as np
import matplotlib.pyplot as plt
import time

# list of the different quadrature order in Lebedev
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
              ["npoints_fibonacci",     "",           300, [1, 1e6],   "int", 
                        "Number of points on the sphere for the Fibonacci integration"],
              ["norder_index",     "",        20, [0, 31],   "int",       
                  "Order index for the Lebedev quadrature"],
              ["orderlist",     "",        65, [3, 131],   "int",
                        "List of the different quadrature order in Lebedev"],
              ["npoints_gauss",     "",        150, [0, 10e4],   "int",
                "points for each angle for the Lebedev quadrature (ntotal = npoints**2)"],

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


def Iq_fibonacci(q, sld, sld_solvent, nsides:int, Rave, L, npoints_fibonacci:int=1000):
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

    # Génération des directions
    q_unit,w = fibonacci_sphere(npoints_fibonacci)   # shape (npoints, 3)

    # Construction des projections
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Calcul des intensités élémentaires
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # Moyenne uniforme sur la sphère
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)
    integral = np.mean(intensity, axis=1)
    time_end = time.time()
    time_fib = time_end - time_start
   
    print(f'Execution time Fibonacci with {npoints_fibonacci} points: {time_fib:.4f} seconds')

    return integral * (sld - sld_solvent)**2, time_fib, n_points_total


Iq_fibonacci.vectorized = True




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

    order = int(orderlist[norder_index])
    q_unit, w = leblib.get_points_and_weights(order)  # shape (npoints,3)

    return q_unit, w


def plot_lebedev_sphere(norder_index, figsize=(7,7)):
    """
    3D representation of Lebedev points on the unit sphere.
    Parameters
    ----------
    norder : int
        Index in orderlist for the Lebedev order.
    """
    pts, w = lebedev_sphere(norder_index)  # shape (npoints,3)
    order = int(orderlist[norder_index])
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



def Iq_lebedev(q, sld, sld_solvent, nsides:int, Rave, L, norder_index:int):
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

    # Iqabc need to have qa, qb and qc with the same shape
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # sum over the points of Lebedev quadrature (axis=1) for each q
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)
    time_end = time.time()
    time_lebedev = time_end - time_start

    print(f'Execution time Lebedev with {len(q_unit)} points: {time_lebedev:.4f} seconds')

    return integral * (sld - sld_solvent)**2, time_lebedev, n_points_total

Iq_lebedev.vectorized = True  # Iq can be vectorized




################################################# Gauss Legendre integration ##################################################


def gauss_legendre_sphere(npoints_gauss: int):
    from sasmodels.gengauss import gengauss
    from numpy.polynomial.legendre import leggauss 
    """
    Generates n_points of Gauss-Legendre on the unit sphere in Cartesian coordinates (x,y,z) 
    and their associated weights.
    Parameters
    ----------
    n_points : int
        Number of Gauss-Legendre points to generate per angle (total = n_points**2).

    Returns
    -------
    points : ndarray, shape (n_points**2, 3)
        Cartesian coordinates of the points on the unit sphere.
    weights : ndarray, shape (n_points**2,)
        Weights associated with each point for integration over the sphere.

    """
    # If we want to use the file generated by gengauss.py: not working right now
    # Retreive the points and weights from the file generated by gengauss.py
    #gengauss(n_points, path) # génère le fichier gauss(n_points).c dans le répertoire path
    # file_gauss = path + f"/gauss{n_points}.c" # retreive the file
    
    # Instead we use leggauss from numpy (was also done in gengauss.py)
    GaussZ, GaussWt = leggauss(npoints_gauss) # voir temps d'exécution
    # GaussZ, GaussWt = np.loadtxt(file_gauss, unpack=True, comments='//')  # charge les points et poids

    # θ : Gauss-Legendre on [0, pi]
    z_theta, w_theta = GaussZ, GaussWt
    theta = 0.5 * (z_theta * (np.pi - 0) + 0 + np.pi) # from [-1,1] to [0,pi], angle = 0.5 * (z * (b - a) + (a + b))
    weights_theta = 0.5 * (np.pi - 0) * w_theta # formula : w = 0.5 * (b - a)

    # φ : Gauss-Legendre on [0, 2pi]
    z_phi, w_phi = GaussZ, GaussWt
    # Linear transformation for phi
    phi = np.linspace(0, 2*np.pi, npoints_gauss, endpoint=False)
    weights_phi = np.full(npoints_gauss, 2*np.pi / npoints_gauss)

    # Grille (θ, φ) :  grid of npoints_gauss x npoints_gauss
    # for each point of θ, we associate all points of φ and vice versa
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    w_theta_grid, w_phi_grid = np.meshgrid(weights_theta, weights_phi, indexing='ij') 

    # Coordonnées x, y, z
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    # Poids totaux
    weights = w_theta_grid * w_phi_grid * np.sin(theta_grid)/(4*np.pi)  # sin(theta) comes from the Jacobian in spherical coordinates
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())) # ravel() to convert multi dimensional array to 1D array
    weights = weights.ravel() # multi dimensional array to 1D array


    return points, weights

def plot_gauss_sphere(npoints_gauss, figsize=(7,7)):
    """
    3D representation of Gauss-Legendre points on the unit sphere.
    Parameters
    ----------
    npoints_gauss : int
        Number of points to generate and display.
    
    """
    pts, w = gauss_legendre_sphere(npoints_gauss)  # shape (npoints,3)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Gauss-Legendre points on the unit sphere, ({len(pts)} points in total)")
    ax.set_box_aspect([1,1,1])  
    plt.show()  

def Iq_gauss_legendre(q, sld, sld_solvent, nsides:int, Rave, L, npoints_gauss:int):
    """
    Integration over the unit sphere with Gauss-Legendre points.

    Parameters
    ----------
    q : float ou array
        Norm of the scattering vector
    sld, sld_solvent : density contrasts
    nsides, Rave, L : geometric parameters of the nanoprism
    npoints_gauss : int
        Number of Gauss-Legendre points per dimension (total npoints_gauss**2)
    Returns
    -------
    Iq : ndarray
        Scattering intensity averaged over all orientations
    time_gauss : float
        Execution time of the function in seconds
    n_points_total : int
        Total number of points used in the quadrature (npoints_gauss**2)
    """

    time_start = time.time()
    nsides = int(nsides)
    q = np.atleast_1d(q)  # make sure q is a 1D table
    n_points_total = npoints_gauss**2


    q_unit, w = gauss_legendre_sphere(npoints_gauss)   # shape (npoints, 3)

    # Projections
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Compute individual intensities
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # Uniform average over the sphere
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)
    time_end = time.time()
    time_gauss = time_end - time_start

    print(f'Execution time Gauss-Legendre with {npoints_gauss**2} points: {time_gauss:.4f} seconds')

    return integral * (sld - sld_solvent)**2, time_gauss, n_points_total

Iq_gauss_legendre.vectorized = True  # Iq can be vectorized






####################################### Common functions #####################################################

def compare_fibonacci_lebedev_spheres(n_fibonacci, norder_lebedev_index, n_gauss_legendre):
    """
    Compare the distribution of points on the unit sphere for the three quadratures: Fibonacci, Lebedev and Gauss-Legendre.
    3D representation of the points on the unit sphere for each quadrature.
    Each quadrature has a different definition of number of points:
    - n_fibonacci : total number of points on the unit sphere
    - norder_lebedev_index : index in orderlist for the Lebedev order
    - n_gauss_legendre : number of points per angle (total n_gauss_legendre**2)
    
    Parameters
    ----------
    n_fibonacci : int
        total number of points on the unit sphere
    norder_lebedev : int
        index in orderlist for the Lebedev order
    n_gauss_legendre : int
        number of points per angle (total n_gauss_legendre**2)
    """

    order = int(orderlist[norder_lebedev_index])
    pts_fib, w_fib = fibonacci_sphere(n_fibonacci)
    pts_leb, w_leb = lebedev_sphere(norder_lebedev_index)
    pts_gauss, w_gauss = gauss_legendre_sphere(n_gauss_legendre)

    print("Sum of the weights Gauss-Legendre :", np.sum(w_gauss))
    print("Sum of the weights Fibonacci :", np.sum(w_fib))
    print("Sum of the weights Lebedev :", np.sum(w_leb))

    ### now plot them in a same plot in 3D
    fig2 = plt.figure(figsize=(8,5))
    ax = fig2.add_subplot(111, projection="3d")
    ax.scatter(pts_gauss[:,0], pts_gauss[:,1], pts_gauss[:,2], s=10, alpha=0.6, label=f"Gauss-Legendre ({len(pts_gauss)} points)")
    ax.scatter(pts_fib[:,0], pts_fib[:,1], pts_fib[:,2], s=10, alpha=0.6, label=f"Fibonacci ({len(pts_fib)} points)")
    ax.scatter(pts_leb[:,0], pts_leb[:,1], pts_leb[:,2], s=10, alpha=0.6, label=f"Lebedev (order of the polynom: {order}, {len(pts_leb)} points)")
    ax.set_box_aspect([1,1,1])
    ax.legend()
    plt.show()
