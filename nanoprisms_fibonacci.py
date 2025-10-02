r"""
This model ...

References
----------


Authorship and Verification
---------------------------

* **Authors:** Jules Marcone, Marianne Imperor-Clerc **Date:** 07/09/2025
* **Last Modified by:** Nicolas Ratel-Ramond **Date:** 02/10/2025
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
              ["n_order", "", 20, [0, inf], "",
               "number for Lebedev order in orderlist"]
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


def Iq_lebedev(q, sld, sld_solvent, nsides:int, Rave, L, norder:int):
    nsides = int(nsides)
    norder = int(norder)
    order = int(orderlist[norder])
    q_unit, w = leblib.get_points_and_weights(order)  # shape (npoints,3)

    q = np.atleast_1d(q)  # make sure q is a 1D table
    # qa, qb, qc have the shape (nq, npoints)
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Iqabc need to have qa, qb and qc with the same shape
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # sum over the points of Lebedev quadrature (axis=1) for each q
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)

    return integral * (sld - sld_solvent)**2



def fibonacci_sphere(npoints: int):
    """
    Génère npoints répartis quasi-uniformément sur la sphère unité
    en coordonnées cartésiennes (x,y,z).
    """
    indices = np.arange(0, npoints, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/npoints)        
    theta = 2 * np.pi * indices / ((1 + 5**0.5) / 2)  
    
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    weight = np.full(len(x),1/npoints)
    
    return np.column_stack((x, y, z)),weight

def plot_fibonacci_sphere(npoints=500):
    """
    Représentation 3D des points de Fibonacci sur la sphère unité.
    Paramètre
    ---------
    npoints : int
        Nombre de points à générer et afficher.
    """
    pts,w = fibonacci_sphere(npoints)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10, alpha=0.6)

    # Axes et esthétique
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Points de Fibonacci sur la sphère unité ({npoints} points)")
    ax.set_box_aspect([1,1,1])  # sphère non déformée
    plt.show()

def Iq_fibonacci(q, sld, sld_solvent, nsides:int, Rave, L, npoints:int=1000):
    """
    Intégration orientationnelle sur la sphère unité avec points de Fibonacci.
    Chaque point a un poids identique = 1/npoints.
    
    Paramètres
    ----------
    q : float ou array
        Module du vecteur de diffusion
    sld, sld_solvent : contrastes de densité de diffusion
    nsides, Rave, L : paramètres géométriques du prisme
    npoints : int
        Nombre de points de Fibonacci sur la sphère
    """
    nsides = int(nsides)
    q = np.atleast_1d(q)  # vecteur q

    # Génération des directions
    q_unit,w = fibonacci_sphere(npoints)   # shape (npoints, 3)

    # Construction des projections
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Calcul des intensités élémentaires
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # Moyenne uniforme sur la sphère
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)
    integral = np.mean(intensity, axis=1)

    return integral * (sld - sld_solvent)**2

## Fonction globale qui laisse le choix de la quadrature en option: 
def Iq(q, sld, sld_solvent, nsides:int, Rave, L, 
       quadrature:str="lebedev", norder:int=20, npoints:int=1000):
    """
    Intégration orientationnelle sur la sphère unité pour calculer l’intensité.
    
    Paramètres
    ----------
    q : float ou array
        Module du vecteur de diffusion
    sld, sld_solvent : contrastes de densité de diffusion
    nsides, Rave, L : paramètres géométriques du prisme
    quadrature : str
        Méthode d’intégration ('lebedev' ou 'fibonacci')
    norder : int
        Ordre pour la quadrature de Lebedev (index dans orderlist)
    npoints : int
        Nombre de points pour la grille de Fibonacci
    """
    nsides = int(nsides)
    q = np.atleast_1d(q)

    if quadrature.lower() == "lebedev":
        norder = int(norder)
        order = int(orderlist[norder])
        q_unit, w = leblib.get_points_and_weights(order)  # shape (npoints, 3)

    elif quadrature.lower() == "fibonacci":
        q_unit, w = fibonacci_sphere(npoints)  # shape (npoints, 3)

    else:
        raise ValueError("quadrature must be 'lebedev' or 'fibonacci'")

    # Construction des projections
    qa = q[:, np.newaxis] * q_unit[:, 0][np.newaxis, :]
    qb = q[:, np.newaxis] * q_unit[:, 1][np.newaxis, :]
    qc = q[:, np.newaxis] * q_unit[:, 2][np.newaxis, :]

    # Calcul des intensités élémentaires
    intensity = Iqabc(qa, qb, qc, nsides, Rave, L)  # shape (nq, npoints)

    # Intégration pondérée sur la sphère
    integral = np.sum(w[np.newaxis, :] * intensity, axis=1)

    return integral * (sld - sld_solvent)**2

Iq.vectorized = True



Iq_fibonacci.vectorized = True
Iq_lebedev.vectorized = True  # Iq can be vectorized
