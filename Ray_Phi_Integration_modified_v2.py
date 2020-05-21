from fipy import FaceVariable, CellVariable, Gmsh2DIn3DSpace, VTKViewer, TransientTerm, ExplicitDiffusionTerm, DiffusionTerm, ExponentialConvectionTerm, DefaultSolver #, getCellVariableDatapoint
from fipy.variables.variable import Variable
from fipy.tools import numerix
from mpl_toolkits.mplot3d import Axes3D
import time
import pickle
from shutil import copyfile
import vtk
import numpy as np
import matplotlib
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
import ray

def get_rho_tuple(number_of_cells):
    #Filename = '/home/debasis/MEGA/My_Github_repositories/FP_MRAM_FiPy_stable/Fresh_start/90_degree/Uniaxial_field_only_uniform_rho/with_axis_0_VTK_files/with_axis_0_img_00000.vtk' #This file contains the phi value
    Filename = '../with_axis_0_img_00401.vtk'
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(Filename)
    reader.ReadAllScalarsOn()
    reader.Update()
    usg = reader.GetOutput().GetCellData().GetScalars()
    z = []
    for i in range(number_of_cells):
        z.append(usg.GetTuple(i)[0])
    return tuple(z)

##### Internal functions
@ray.remote
def task_loop(worker_idx, phi_array, theta_array, rho):
    # Each worker will iterate to emulate a new worker working on a new task
    # Done this way to minimize number of actual task creations and
    # destructions which will slow down overall execution
    global_idx = worker_idx
    worker_accum = 0
    nth = len(theta_array)
    nphi = len(phi_array)
    total_tasks=(nth-1)*(nphi-1)
    while global_idx < total_tasks:
        # Pick up range over which the task is supposed to perform integration

        #### First, convert global_idx to the indices of theta and phi
        local_phi_idx, local_theta_idx = divmod(global_idx, nth)

        #### Get range of integration using indices
        min_phi = phi[local_phi_idx]
        max_phi = phi[local_phi_idx+1]
        min_theta = theta[local_theta_idx]
        max_theta = theta[local_theta_idx+1]
        #print('----------------------------------------------------------')
        #conv_fac=(180.0/numerix.pi)
        #print(global_idx)
        #print('Phi_idx = ' + str(local_phi_idx))
        #print('Theta_idx = ' + str(local_theta_idx))
        #print('Phi = ' + str(min_phi*conv_fac) + ' to ' + str(max_phi*conv_fac))
        #print('Theta_idx = ' + str(min_theta*conv_fac) + ' to ' + str(max_theta*conv_fac))
        #print('----------------------------------------------------------')

        # Calculate the integral
        local_probVal, local_error = dblquad(lambda y, x: (numerix.sin(y) * rho([[numerix.sin(y)*numerix.cos(x)], [numerix.sin(y)*numerix.sin(x)], [numerix.cos(y)]], order=1)), min_phi, max_phi, lambda phix: min_theta, lambda phix: max_theta, epsabs=1e-15, epsrel=1e-8)

        # Accumulate local integral into worker track of sum
        worker_accum = worker_accum + local_probVal

        # Increment global index
        global_idx = global_idx + num_workers
    # Return worker accumulator result
    return worker_accum

if __name__ == '__main__':
    plt.rcParams.update({'font.size':20})

    num_workers=24
    ray.init(num_cpus=num_workers,memory=102005473280,object_store_memory=21474836480)

    #### Load mesh details from a saved file

    mesh=pickle.load(open("../mesh_details_cellsize_0pt008_extrude_1pt00001.p","rb"))
    gridCoor = mesh.cellCenters
    mUnit = gridCoor
    mNorm = numerix.linalg.norm(mUnit,axis=0)
    print('max mNorm='+str(max(mNorm)))
    print('min mNorm='+str(min(mNorm)))
    mAllCell = mUnit / mNorm

    msize=numerix.shape(mAllCell)
    number_of_cells=msize[1]
    print('Total number of cells = ' +str(number_of_cells))

    #### Loading phi values from a saved file

    rho_value = get_rho_tuple(number_of_cells) # get phi values from a previous .vtk file
    rho = CellVariable(name=r"$\Rho$",mesh=mesh,value=rho_value)


    ##########################################################################
    #                                    Check from here                                                                                 #
    ##########################################################################

    nth=180001
    nphi=90001
    total_tasks=(nth-1)*(nphi-1)
    theta=numerix.linspace(0,numerix.pi,nth)
    phi=numerix.linspace(0,0.5*numerix.pi,nphi)

    # Load shared memory
    theta_id = ray.put(theta)
    phi_id = ray.put(phi)
    rho_id = ray.put(rho)

    task_result = []
    for worker_idx in range(num_workers-1):
        task_result.append(task_loop.remote(worker_idx, phi_id, theta_id, rho_id))
        #task_result.append(task_loop(worker_idx, phi, theta, rho))

    prob_array = ray.get(task_result)
    prob_result = numerix.sum(prob_array)

    #prob_result = numerix.sum(task_result)
