r"""
This model ...

References
----------


Authorship and Verification
---------------------------

* **Author:** Steve King **Date:** 19/11/2019
* **Last Modified by:** Steve King **Date:** 24/06/2020
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

def Iqabc(qa,qb,qc,nsides,Rave,L):
    nsides=int(nsides)
    edge = functions.edge_from_gyration_radius(nsides,Rave)
    intensity=formfactor.I_nanoprism([qa,qb,qc],nsides,edge,L)/(functions.volume_nanoprism(nsides,edge,L))**2
    return intensity

def Iq(q,sld,sld_solvent,nsides:int,Rave,L,norder:int):
    nsides=int(nsides)
    norder=int(norder)
    "Lebedev integration at q"
    order=int(orderlist[norder])
    q_unit,w = leblib.get_points_and_weights(order)
    qa=q*q_unit[:,0]
    qb=q*q_unit[:,1]
    qc=q*q_unit[:,2]
    integral = 1 * np.sum(w * Iqabc(qa,qb,qc,nsides,Rave,L))
    return integral*(sld-sld_solvent)**2

Iq.vectorized = False  # Iq only for one float value q
