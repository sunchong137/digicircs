#!/usr/bin/env python
'''
Draw a circuit with the given circuit string.
Author: Chong Sun <sunchong137@gmail.com>
'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
from   scipy.interpolate    import griddata
import matplotlib.colors    as col
from matplotlib import collections  as mc
from matplotlib.patches import Rectangle

plt.rc('font',family='serif')
plt.rc('xtick',labelsize='large')
plt.rc('ytick',labelsize='large')
plt.rc('legend',fontsize='large')
plt.rc('lines', linewidth=2)
plt.rc('savefig', dpi=400)
from matplotlib import rcParams
rcParams.update({'figure.autolayout':True})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


Pauli_2q = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
ctrl_2q = {"CNOT":"X", "CRX":"RX", "CRY": "RY", "CRZ": "RZ"}

# TODO:  Fix the text size with the following suggestion
# https://stackoverflow.com/questions/39490274/matplotlib-fontsize-in-terms-of-axis-units

def draw_wires(ax, n_qubits, n_layers, zorder=0, tshift=0.08):
    '''
    draw wires based on the number of qubits and layers.
    '''
    x = np.linspace(0-0.5, n_layers+0.5, 10)
    for i in range(n_qubits):
        y = np.ones(10) * i
        ax.plot(x, -y, c='k', lw=1, zorder=zorder)
        ax.text(-0.8, -i-tshift, r"$|0\rangle$", fontsize=20)

def rectang(ax, x, y, lx, ly, zorder=1):
    x_vert = np.array([x-lx/2, x+lx/2])
    y_vert = np.array([y-ly/2, y+ly/2])
    _i = np.ones(2)
    ax.plot(_i*x_vert[0], y_vert, lw=1, c='k',zorder=zorder)
    ax.plot(_i*x_vert[1], y_vert, lw=1, c='k',zorder=zorder)
    ax.plot(x_vert, _i*y_vert[0], lw=1, c='k',zorder=zorder)
    ax.plot(x_vert, _i*y_vert[1], lw=1, c='k',zorder=zorder)
    ax.fill_between(x_vert, y_vert[0]*_i, y_vert[1]*_i, color='w')



def draw_gate(ax, i_layer, t_qubit, c_qubit=None, t_symb=None, c_symb=None, zorder=1,
              box_size=0.5, lsize=20, tshift=0.08):
    '''
    draw a one-qubit or two-qubit gate.
    '''
    t_qubit = -t_qubit
    # draw gate on target
    rectang(ax, i_layer, t_qubit, box_size, box_size)
    if t_symb is not None:
        if len(t_symb) == 2:
            x_shift = tshift*2
        else:
            x_shift = tshift
        ax.text(i_layer-x_shift, t_qubit-tshift, t_symb, fontsize=lsize)

    # draw control
    if c_qubit is not None:
        c_qubit = -c_qubit
        if abs(t_qubit) < abs(c_qubit):
            t_y = t_qubit - box_size/2
        else:
            t_y = t_qubit + box_size/2
        if c_symb is None:
            ax.scatter([i_layer], [c_qubit], c='k', s=30)
            c_y = c_qubit
        else:
            rectang(ax, i_layer, c_qubit, box_size, box_size)
            #rect2 = Rectangle((i_layer, c_qubit), box_size, box_size, lw=1, ec='k', zorder=zorder)
            #ax.add_patch(rect)
            ax.text(i_layer-tshift, c_qubit-tshift, c_symb, fontsize=lsize)
            if abs(t_qubit) < abs(c_qubit):
                c_y = c_qubit + box_size/2
            else:
                c_y = c_qubit - box_size/2
        x = np.ones(2)*i_layer
        y = np.array([c_y, t_y])
        ax.plot(x, y, c='k', lw=1)


def draw_circuit(q_str, n_qubit, save_name=None):

    q_list = q_str.split("@")
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    #fig.set_size_inches(6,4)
    # draw gates
    qubit_count = []
    n_layer = 0
    for gate in q_list:
        g_info = gate.split("=")
        g_name = g_info[0]
        t_qbit = int(g_info[1])
        c_qbit = g_info[2]
        if c_qbit != "nop":
            c_qbit = int(c_qbit)
        else:
            c_qbit = None
        if g_name in Pauli_2q:
            t_symb = g_name[1]
            c_symb = g_name[0]
        elif g_name in ctrl_2q:
            t_symb = ctrl_2q[g_name]
            c_symb = None
        else:
            t_symb = g_name
            c_symb = None

        if c_qbit is not None:
            qmin = min(c_qbit, t_qbit)
            qmax = max(c_qbit, t_qbit)
            for i in range(qmin, qmax + 1):
                if i in qubit_count: # start a new layer
                    n_layer += 1
                    qubit_count = []
                    break
            qubit_count += list(range(qmin, qmax + 1))
        else:
            if t_qbit in qubit_count: # start a new layer
                n_layer += 1
                qubit_count = []
            qubit_count += [t_qbit]

        draw_gate(ax, n_layer, t_qbit, c_qubit = c_qbit, t_symb=t_symb, c_symb=c_symb)

    draw_wires(ax, n_qubit, n_layer, zorder=0)
    if save_name is not None:
        fig.savefig(save_name, dpi=300)
    else:
        plt.show()

##################################
# plot functions for deep dreaming
##################################
def closefig():
    """Clears and closes current instance of a plot."""
    plt.clf()
    plt.close()

def running_avg_test_loss(avg_test_loss, directory):
    """Plot running average test loss"""

    plt.figure()
    plt.plot(avg_test_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Running average test loss')
    name = name = directory + '/runningavg_testloss'
    plt.savefig(name)
    closefig()

def test_model_after_train(calc_train, real_vals_prop_train,
               calc_test, real_vals_prop_test,
               directory, prop_name='logP'):
    """Scatter plot comparing ground truth data with the modelled data";
    includes both test and training data."""

    plt.figure()
    plt.scatter(calc_train,real_vals_prop_train,color='red',s=40, facecolors='none')
    plt.scatter(calc_test,real_vals_prop_test,color='blue',s=40, facecolors='none')
    plt.xlim(min(real_vals_prop_train)-0.5,max(real_vals_prop_train)+0.5)
    plt.ylim(min(real_vals_prop_train)-0.5,max(real_vals_prop_train)+0.5)
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    plt.title('Train set (red), test set (blue)')
    name = directory + '/test_model_after_training'
    plt.savefig(name)
    closefig()


def test_model_before_dream(trained_data_prop, computed_data_prop,
                            directory, prop_name='logP'):
    """Scatter plot comparing ground truth data with modelled data"""

    plt.figure()
    plt.scatter(trained_data_prop, computed_data_prop)
    plt.xlabel('Modelled '+prop_name)
    plt.ylabel('Computed '+prop_name)
    name = directory + '/test_model_before_dreaming'
    plt.savefig(name)
    plt.show()
    closefig()


def prediction_loss(train_loss, test_loss, directory):
    """Plot prediction loss during training of model"""

    plt.figure()
    plt.plot(train_loss, color = 'red')
    plt.plot(test_loss, color = 'blue')
    plt.title('Prediction loss: training (red), test (blue)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    name = directory + '/predictionloss_test&train'
    plt.savefig(name)
    closefig()


def dreamed_histogram(prop_lst, prop, directory, prop_name='logP'):
    """Plot distribution of property values from a given list of values
    (after transformation)"""

    plt.figure()
    plt.hist(prop_lst, density=True, bins=30)
    plt.ylabel(prop_name+' - around '+str(prop))
    name = directory + '/dreamed_histogram'
    plt.savefig(name)
    closefig()


def initial_histogram(prop_dream, directory,
                      dataset_name='QM9', prop_name='logP'):
    """Plot distribution of property values from a given list of values
    (before transformation)"""

    plt.figure()
    plt.hist(prop_dream, density=True, bins=30)
    plt.ylabel(prop_name + ' - ' + dataset_name)
    name = directory + '/QM9_histogram'
    plt.savefig(name)
    closefig()


def plot_transform(target, mol, logP, epoch, loss):
    """Combine the plots for logP transformation and loss over number of
    epochs.
    - target: the target logP to be optimized.
    - logP: the transformation of logP over number of epochs.
    - epoch: all epoch #'s where the molecule transformed when dreaming.
    - loss: loss values over number of epochs.
    """

    full_epoch = []
    full_logP = []
    step = -1
    for i in range(len(loss)):
        if i in epoch:
            step += 1
        full_logP.append(logP[step])
        full_epoch.append(i)

    fig, ax1 = plt.subplots()

    color = '#550000'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('LogP', color=color)
    ax1.plot(full_logP, linewidth=1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    color = '#000055'
    ax2.set_ylabel('Training loss', color=color)
    ax2.plot(loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Target logP = '+str(target))
    plt.tight_layout()
    #plt.savefig('dream_results/{}_{}_transforms.svg'.format(target, loss[len(loss)-1]))

    plt.show()

##q_str = "H=0=nop=nop@XY=2=1=nop@CNOT=2=0=nop@CRX=1=0=nop@H=0=1=nop"
##q_str = "H=1=nop=nop@RX=2=0=nop"
#q_str = "Rz=0=nop=nop@Rz=4=nop=nop@CNOT=2=3=nop@Rz=1=nop=nop@Rz=5=nop=nop@CNOT=0=2=nop@CNOT=3=4=nop"
#draw_circuit(q_str, 6)
