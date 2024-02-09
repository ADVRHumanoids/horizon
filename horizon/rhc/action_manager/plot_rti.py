import h5py
import numpy as np
import matplotlib.pyplot as plt
from horizon.utils.plotter import createPlotGrid
f = h5py.File('ciao__0_2023_02_21__15_05_52.mat', 'r')



solution = dict()
for key in f.keys():
    solution[key] = np.array(f[key])

contacts = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']

pos_xy_fig, pos_xy_gs = createPlotGrid(2, len(contacts), f'feet position - plane_xy')
pos_xz_fig, pos_xz_gs = createPlotGrid(2, len(contacts), f'feet position - plane_xz')
f_fig, f_gs = createPlotGrid(2, len(contacts), f'contact forces')

for i, contact in enumerate(contacts):

    # assuming that cartesian position of the contact exists and is called contact + 'pos'
    pos = solution[contact + 'pos'].T

    ax_xy = pos_xy_fig.add_subplot(pos_xy_gs[i])
    ax_xy.grid()

    ax_xy.plot(pos[0, :], pos[1, :], linewidth=2.5)
    ax_xy.scatter(pos[0, 0], pos[1, 0])
    ax_xy.scatter(pos[0, -1], pos[1, -1], marker='x')
    ax_xy.set_title(contact)
    ax_xy.label_outer()

    # ==================================================

    ax_xz = pos_xz_fig.add_subplot(pos_xz_gs[i])
    ax_xz.grid()

    ax_xz.plot(pos[0, :], pos[2, :], linewidth=2.5)
    ax_xz.scatter(pos[0, 0], pos[2, 0])
    ax_xz.scatter(pos[0, -1], pos[2, -1], marker='x')
    ax_xz.set_title(contact)
    ax_xz.label_outer()

    # ==================================================

    ax_f = f_fig.add_subplot(f_gs[i])
    ax_f.grid()

    # assuming that force of the contact exists and is called 'f' + contact
    force = solution['f_' + contact]
    ax_f.plot(force)
    ax_f.set_title(contact)
    ax_f.label_outer()

plt.show()







