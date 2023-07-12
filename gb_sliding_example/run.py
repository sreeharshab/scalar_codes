from pipelines import *
import ase
import numpy as np
from ase.io import read, write
from ase.calculators.vasp import Vasp
from matplotlib import pyplot as plt

if __name__=='__main__':
    atoms = read("POSCAR")
    opt_levels = {
        1: {"kpts": [1,3,3], "ismear": 0, "sigma": 0.02, "amin": 0.01},
        2: {"kpts": [1,7,5], "amin": 0.01},
        3: {"kpts": [1,15,9], "amin": 0.01},
    }
    n = 11
    n_steps = 20
    disp = (3.8663853050000001/5)   # Remember to adjust this.
    theta = (0*pi)/180
    system = slide_sigma3_gb(n_steps)
    
    # Running simulation
    system.run_serial(atoms, opt_levels, n, disp, theta, scheme="step", restart=True)

    # Saving the output trajectory
    system.get_output_Trajectory(atoms, theta, calc_type='serial')
    
    # Obtaining the energies
    E = system.analysis(theta, property="Energy")
    step = np.arange(n_steps+1) # +1 is done to consider the initial non-slided GB.
    fig = plt.figure(dpi = 200, figsize=(4.5,4))
    plt.plot(step[1:], E, color = 'indigo')
    plt.scatter(step[1:], E, s = 10, color = 'indigo')
    get_plot_settings(fig, x_label="Step",y_label="Energy (eV/atom)",fig_name=f"./{int((theta/pi)*180 + 0.1)}/Evsstep")

    # Layer movement
    coord = np.zeros((1,n_steps+1,2))
    indices = np.array([49, 52, 56, 61, 65, 69, 73, 76, 80, 85, 4, 87, 7, 10, 14, 19, 24, 28, 31, 34, 38, 43])
    for index in indices:
        tmp = system.get_layer_movement(n, theta, index = index)
        coord = np.append(coord, tmp[np.newaxis,:,:], axis = 0)
    coord = np.delete(coord, 0, 0)
    
    layer_10_disp = np.array([])
    layer_11_disp = np.array([])
    
    fig = plt.figure(dpi = 200, figsize=(4.5,4))
    cmap = plt.get_cmap('viridis')
    values = np.linspace(0,1,21)
    for i in range(n_steps+1):
        y = np.array([])
        z = np.array([])
        # This works for 0 degree sliding
        for j in range(indices.size):
            if (i==4) and (j > 11 and j < 17):
                y = np.append(y, coord[j,i,0]+3.8663853)
            elif (i==5) and (j > 10 and j < 19):
                y = np.append(y, coord[j,i,0]+3.8663853)
            elif (i==6) and (j > 10 and j < 21):
                y = np.append(y, coord[j,i,0]+3.8663853)
            elif (i>=7 and i <=8) and (j > 10 and j < 22):
                y = np.append(y, coord[j,i,0]+3.8663853)
            elif (i==9) and (j > 11 and j < 17):
                y = np.append(y, coord[j,i,0]+7.7327706)
            elif (i==9) and (j ==0 or j == 11 or j >= 17):
                y = np.append(y, coord[j,i,0]+3.8663853)
            elif (i==10) and (j > 10 and j < 18):
                y = np.append(y, coord[j,i,0]+7.7327706)
            elif (i==10) and (j ==0 or j >= 18):
                y = np.append(y, coord[j,i,0]+3.8663853)
            elif (i==11) and (j > 10 and j < 21):
                y = np.append(y, coord[j,i,0]+7.7327706)
            elif (i==11) and (j >= 21):
                y = np.append(y, coord[j,i,0]+3.8663853)
            else:
                y = np.append(y, coord[j,i,0])
            z = np.append(z, coord[j,i,1])
        x = np.arange(0,indices.size,1)

        ## Calculating the displacement of the layer from its initial configuration
        layer_disp = np.array([])
        for j in range(indices.size):
            tmp = ((y[j] - coord[j,0,0])**2)**(1/2)
            layer_disp = np.append(layer_disp, tmp)

        ## Displacement of 10th and 11th layer
        layer_10_disp = np.append(layer_10_disp, layer_disp[10])
        layer_11_disp = np.append(layer_11_disp, layer_disp[11])

        ## Plotting 1
        plt.plot(x,layer_disp, color = cmap(values[n_steps-i]))
        plt.scatter(x,layer_disp, s = 10, color = cmap(values[n_steps-i]))
        plt.text(x[16], layer_disp[16], f"{i}", fontsize = 6, position = (x[16]+0.1, layer_disp[16]+0.1))
    get_plot_settings(fig, x_label="Layer",y_label="Displacement (Å)",fig_name=f"./{int((theta/pi)*180 + 0.1)}/p.png")
    
    ## Plotting 2
    fig = plt.figure(dpi = 200, figsize=(4.5,4))
    plt.plot(step, layer_10_disp, color = "firebrick", label = "Layer 10")
    plt.scatter(step, layer_10_disp, s = 10, color = "firebrick")
    # plt.text(step[20], layer_10_disp[20], "10", fontsize = 6, position = (step[16]-0.8, layer_10_disp[16]+0.1))
    plt.plot(step, layer_11_disp, color = "maroon", label = "Layer 11")
    plt.scatter(step, layer_11_disp, s = 10, color = "maroon")
    # plt.text(step[20], layer_11_disp[20], "11", fontsize = 6, position = (step[16]-0.8, layer_11_disp[16]+0.1))
    get_plot_settings(fig, x_label="Step",y_label="Displacement of GB Layers (Å)",fig_name=f"./{int((theta/pi)*180 + 0.1)}/disp.png")