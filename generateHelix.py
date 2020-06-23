import numpy as np 
import matplotlib.pyplot as plt
from geomdl import fitting
from geomdl import convert
from geomdl import construct
from geomdl.visualization import VisMPL as vis
import sys
np.set_printoptions(threshold=sys.maxsize)
from tools import preProcess
from geomdl import exchange
from geomdl import knotvector
from geomdl import BSpline 

def helix():

    # create starting paramters for helix
    M = 8
    t = np.linspace(0,2*np.pi/3,M)
    a = 30
    b = 30

    s = []
    C = a**2+b**2
    for i in range(0,len(t)):
        s.append(np.sqrt(C)*t[i])

    r = []
    T = []
    N = []
    B = []

    # create the tangential, normal, and binormal vectors
    for i in range(0,len(s)):
        r.append([a*np.cos(s[i]/np.sqrt(C)),a*np.sin(s[i]/np.sqrt(C)),b*s[i]/np.sqrt(C)])
        T.append([-a/np.sqrt(C)*np.sin(s[i]/np.sqrt(C)),a/np.sqrt(C)*np.cos(s[i]/np.sqrt(C)),b/np.sqrt(C)])
        N.append([-np.cos(s[i]/np.sqrt(C)),-np.sin(s[i]/np.sqrt(C)),0])

    B.append(np.cross(T,N))

    r = np.array(r)
    # print(r)

    # scale the vectors up by a factor of 20
    Ts = np.array(T)
    Ns = np.array(N)
    Bs = np.array(B[0])
    # print(Bs)

    # scatter the T, N, and B vectors
    # fig = plt.figure()
    # ax = plt.axes(projection = "3d")
    # # ax.scatter(Ts[:,0],Ts[:,1],Ts[:,2],color = 'r')
    # ax.plot(r[:,0],r[:,1],r[:,2],color = 'r')
    # ax.scatter(r[5,0]+Ts[5,0],r[5,1]+Ts[5,1],r[5,2]+Ts[5,2],color = 'b')
    # ax.scatter(r[5,0],r[5,1],r[5,2],color = 'k')
    # ax.scatter(r[5,0]-Ts[5,0],r[5,1]-Ts[5,1],r[5,2]-Ts[5,2],color = 'g')

    # ax.scatter(Bs[:,0],Bs[:,1],Bs[:,2],color = 'g')
    # ax.scatter(Ns[:,0],Ns[:,1],Ns[:,2],color = 'b')


    helix = []
    phi = np.linspace(0,2*np.pi,M)
    d = 10
    for i in range(0,len(s)):
        for j in range(0,len(phi)):
            helix.append([  d*Bs[i,0]*np.cos(phi[j])+d*Ns[i,0]*np.sin(phi[j])+r[i,0],
                            d*Bs[i,1]*np.cos(phi[j])+d*Ns[i,1]*np.sin(phi[j])+r[i,1],
                            d*Bs[i,2]*np.cos(phi[j])+d*Ns[i,2]*np.sin(phi[j])+r[i,2]
                        ])

    helix = np.array(helix)

    # ax.scatter(helix[:,0],helix[:,1],helix[:,2])
    # plt.show()
    p_ctrlpts = helix
    size_u = M
    size_v = M
    degree_u = 5
    degree_v = 3

    # Do global surface approximation
    surf = fitting.approximate_surface(p_ctrlpts, size_u, size_v, degree_u, degree_v)

    surf = convert.bspline_to_nurbs(surf)

    # Extract curves from the approximated surface
    surf_curves = construct.extract_curves(surf)
    plot_extras = [
        dict(
            points=surf_curves['u'][0].evalpts,
            name="u",
            color="red",
            size=10
        ),
        dict(
            points=surf_curves['v'][0].evalpts,
            name="v",
            color="black",
            size=10
        )
    ]
    tube_pcl = np.array(surf.evalpts)
    tube_cpts = np.array(surf.ctrlpts)

    # np.savetxt("cpts_bezier.dat",r,delimiter = ',')
    # from matplotlib import cm
    surf.delta = 0.02
    surf.vis = vis.VisSurface()
    surf.render(extras=plot_extras)
    # exchange.export_obj(surf, "helix.obj")
    # np.savetxt("RV_tube.dat",tube_pcl,delimiter = ' ')
    # np.savetxt("tube_cpts.dat",tube_cpts,delimiter = ' ')


    # crv = BSpline.Curve()
    # crv.degree = 3
    # crv.ctrlpts = exchange.import_txt("cpts_tube.dat")
    # crv.knotvector = knotvector.generate(crv.degree, crv.ctrlpts_size)

    # # Set the visualization component of the curve
    # crv.vis = vis.VisCurve3D()

    # # Plot the curve
    # crv.render()
    return surf 

