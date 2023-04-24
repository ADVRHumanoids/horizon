import h5py
import matplotlib.pyplot as plt
import numpy as np

file = '/home/fruscelli/Downloads/robot_state_log__0_2023_02_09__16_37_55.mat'

f = h5py.File(file)
print(f.keys())
arm_1 = np.array(f['arm_1']).astype(int).flatten() - 1
arm_2 = np.array(f['arm_2']).astype(int).flatten() - 1
arm_3 = np.array(f['arm_3']).astype(int).flatten() - 1

sec_i = 60000
sec_f = 85000


motor_pos = np.array(f['motor_position'])
link_pos = np.array(f['link_position'])
pos_ref = np.array(f['position_reference'])

motor_vel = np.array(f['motor_velocity'])
link_vel = np.array(f['link_velocity'])

effort = np.array(f['effort'])

t_slice = slice(sec_i, sec_f)
indices = arm_1
np_indices = np.s_[t_slice, indices]
# t_indices = slice(None)
# plt.plot(motor_pos[np_indices])
# plt.plot(link_pos[np_indices])
# plt.plot(pos_ref[np_indices])
# plt.plot(effort[np_indices])
# plt.plot(motor_vel[np_indices])

plt.show()



#
# plt.plot(solution[])