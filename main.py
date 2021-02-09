# imports
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geomdl.visualization import VisMPL as vis
from slice import slice
from tools import preProcess
from geomdl import construct
from geomdl import exchange
import numpy as np
import sys
from test_fitRemappedRV import fit_Remapped_RV
from fitStandardRV import fit_Standard_RV
plt.style.use('seaborn')
sys.path.append("../")


def main():

    path = "d:/Workspace/RV-Mapping/RVshape/"
    rm_file = "rm_RVendo_0"
    std_file = "standardSample_15x25_RVendo_0"
    remapped_RV_file = path + rm_file
    standard_RV_file = path + std_file
    trimmed_rv_file = np.loadtxt("d:/Workspace/RV-Mapping/RVshape/transformed_RVendo_0.txt")
    # trimmed_rv_file = np.loadtxt("d:/Workspace/RV-Mapping.py/processed_RVendo_0.txt")

    std_rv_data = np.loadtxt(standard_RV_file + ".csv", delimiter=",")
    # std_rv_data = preProcess(std_rv_data)
    rm_rv_data = np.loadtxt(remapped_RV_file + ".txt", delimiter="\t")
    rm_rv_data = rm_rv_data[rm_rv_data[:,2] > 1]
    rm_rv_data = rm_rv_data[rm_rv_data[:,2] < max(rm_rv_data[:,2]) - 11]
    # rm_rv_data = preProcess(rm_rv_data)
    points = std_rv_data
    # np.savetxt("trimmed_" + rm_file + ".txt",rm_rv_data)
    
    N = 25
    M = 15
    # surf,sampled_data = fit_Remapped_RV(N,M, rm_rv_data,flag = False)
    surf = fit_Standard_RV(N,M, std_rv_data)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    # ax.scatter(trimmed_rv_file[:, 0],trimmed_rv_file[:, 1],trimmed_rv_file[:, 2])
    ax.scatter(std_rv_data[:, 0],std_rv_data[:, 1],std_rv_data[:, 2], s= 50,color = 'r')
    # ax.scatter(points[:, 0],points[:, 1],points[:, 2])
    # ax.scatter(sampled_data[:, 0],sampled_data[:, 1],sampled_data[:, 2] , color = 'r')
    # ax.scatter3D(0*np.ones((len(points))),0*np.ones((len(points))),points[:, 2],color = 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.axis('off')
    plt.show()

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.scatter(std_rv_data[:, 0],std_rv_data[:, 1],std_rv_data[:, 2], color = 'r')
    # ax.scatter(trimmed_rv_file[:, 0],trimmed_rv_file[:, 1],trimmed_rv_file[:, 2])
    # plt.axis('off')
    # plt.show()
    

    # np.savetxt("sampled_" + str(M) + 'x' + str(N) + '_' + rm_file + ".txt",sampled_data)
    

    # Extract curves from the approximated surface
    surf_curves = construct.extract_curves(surf)
    plot_extras = [
        dict(points=surf_curves["u"][0].evalpts, name="u", color="red", size=5),
        dict(points=surf_curves["v"][0].evalpts, name="v", color="black", size=5),
    ]
    surf.delta = 0.02
    surf.vis = vis.VisSurface()
    surf.render(extras=plot_extras)
    # exchange.export_obj(surf, rm_file + "_fit.obj")
    exchange.export_obj(surf, std_file + "_fit.obj")
    # # visualize data samples, original RV data, and fitted surface
    eval_surf = np.array(surf.evalpts)
    np.savetxt(std_file + "_NURBS_surf_pts.csv",eval_surf,delimiter = ',')
    # np.savetxt(rm_file + "_NURBS_surf_pts.csv",eval_surf,delimiter = ',')
    # # eval_surf = preProcess(eval_surf)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(eval_surf[:,0],eval_surf[:,1],eval_surf[:,2], color = 'r')
    # ax.scatter3D(points[:, 0],points[:, 1],points[:, 2])
    ax.scatter3D(trimmed_rv_file[:, 0],trimmed_rv_file[:, 1],trimmed_rv_file[:, 2])
    # ax.scatter3D(xyz[:, 0],xyz[:, 1],xyz[:, 2])
    # ax.scatter(X[:,0],X[:,1],X[:,2])

    # # ax.scatter(X[:,0],X[:,1],X[:,2])
    cpts = np.array(surf.ctrlpts)
    print(len(cpts))
    # np.savetxt('cpts_'+rm_file+".csv",cpts, delimiter = ',')
    np.savetxt('cpts_'+ std_file + ".csv",cpts, delimiter = ',')
    # fig = plt.figure()
    # ax = plt.axes(projection = "3d")
    # ax.scatter(X[:,0],X[:,1],X[:,2])
    # ax.scatter(cpts[:,0],cpts[:,1],cpts[:,2])
    plt.show()


main()
