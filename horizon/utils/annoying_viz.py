from horizon.utils import utils, resampler_trajectory, mat_storer
from horizon.ros.replay_trajectory import *
import matplotlib.pyplot as plt


# ms = mat_storer.matStorer('annoying.mat')
# ms = mat_storer.matStorer('annoying_ref_withSameValueIG_proximalState.mat') # terrible
ms = mat_storer.matStorer('annoying_ref_withLinearIG_proximalState.mat') # terrible
# ms = mat_storer.matStorer('annoying_ref_withSameValueIG.mat')
# ms = mat_storer.matStorer('annoying_ref_withNoIG.mat')
# ms = mat_storer.matStorer('annoying_ref_withLinearIG.mat')
solution = ms.load()
dt_res = solution['dt'][0][0]
n_nodes = solution['n_nodes'][0][0]
times = solution['times'][0]
times_res = solution['times_res'][0]

contacts_name = ['lf_foot', 'rf_foot', 'lh_foot', 'rh_foot']
f_list = {i: solution[f'force_{i}'] for i in contacts_name}

# resampled solutions
f_list_res = {i: solution[f'force_{i}_res'] for i in contacts_name}

# ====================================================
# ====================================================
# ====================================================
plot_q = True
if plot_q:

    # =================================== q vs q_res ===================================
    plt.figure()
    for dim in range(solution['q'].shape[0]):
        plt.scatter(times, np.array(solution['q'][dim, :]))

    for dim in range(solution['q_res'].shape[0]):
        plt.plot(times_res, np.array(solution['q_res'][dim, :]))
    plt.title('q vs q_res')

    # =================================== f vs f_res ===================================
    plt.figure()
    for contact_name in f_list.keys():
        for dim in range(f_list[contact_name].shape[0]):
            plt.plot(times[:-1], f_list[contact_name][dim, :])
        plt.title(f'f_{contact_name}')
        for dim in range(f_list_res[contact_name].shape[0]):
            plt.plot(times_res[:-1], f_list_res[contact_name][dim, :], '--')
        plt.title(f'f_{contact_name}_ref')

    # ===================================== tau_base_res vs tau_base ===============================
    plt.figure()
    range_plot_tau = range(6)
    for dim in range_plot_tau:
        plt.scatter(times[:-1], solution[f'tau'][dim, :])
        plt.plot(times_res[:-1], solution[f'tau_res'][dim, :])

    # ===============================================================================================
    plt.figure()
    for contact_name in f_list.keys():
        for dim in range(f_list[contact_name].shape[0]):
            plt.plot(range(f_list[contact_name][dim, :].shape[0]), f_list[contact_name][dim, :])
        plt.title(f'f_{contact_name}')
    plt.figure()
    for dim in range(f_list_res[contact_name].shape[0]):
        plt.plot(range(f_list_res[contact_name][dim, :].shape[0]), f_list_res[contact_name][dim, :], '--')
    plt.title(f'f_{contact_name}_ref')

    plt.show()

replay_traj = False
if replay_traj:

    urdffile = '../examples/urdf/spot.urdf'
    urdf = open(urdffile, 'r').read()
    kindyn = cas_kin_dyn.CasadiKinDyn(urdf)

    n_c = 4
    n_q = kindyn.nq()
    n_v = kindyn.nv()
    n_f = 3

    joint_names = kindyn.joint_names()
    if 'universe' in joint_names: joint_names.remove('universe')
    if 'floating_base_joint' in joint_names: joint_names.remove('floating_base_joint')


    import subprocess, os

    path_to_examples = '../examples'
    os.environ['ROS_PACKAGE_PATH'] += ':' + path_to_examples
    subprocess.Popen(["roslaunch", path_to_examples + "/replay/launch/launcher.launch", 'robot:=spot'])
    rospy.loginfo("'spot' visualization started.")

    # replay NEW trajectory
    repl = replay_trajectory(dt_res, joint_names, solution['q_res'], f_list_res, cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED, kindyn)
    repl.sleep(1.)
    repl.replay(is_floating_base=True)

