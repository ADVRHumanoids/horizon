import phase_manager.pymanager as pymanager
import phase_manager.pyphase as pyphase

from horizon.problem import Problem

import numpy as np
import time

'''
1. phase manager should erase item's nodes (or at least handle them) after adding the item
    right now, if the constraint starts with all nodes ON, addingPhase() won't erase item in nodes were it is not active
'''
if __name__ == '__main__':

    ns = 10
    prb = Problem(ns, receding=True)
    var_1 = prb.createVariable('a', 3)
    var_2 = prb.createVariable('b', 3)

    param_1 = prb.createParameter('param_1', 3)

    cnsrt_1 = prb.createConstraint('cnsrt_1', var_1 * var_2, [])
    cnsrt_2 = prb.createConstraint('cnsrt_2', var_1 - param_1, [])

    cost_1 = prb.createCost('cost_1', var_1 + var_2, [])





    pm = pymanager.PhaseManager(ns)
    # phase manager handling
    timeline_1 = pm.addTimeline(f'timeline_1')


    phase_1_duration = 5
    phase_1 = pyphase.Phase(phase_1_duration, 'phase_1')
    # phase_1.addItem(cnsrt_1)
    # phase_1.addParameterValues(param_1, np.array([[1, 1, 1]] * phase_1_duration).T)
    # phase_1.addCost(cost_1)
    phase_1.addVariableBounds(var_1, np.array([[1, 1, 1]] * phase_1_duration).T, np.array([[1, 1, 1]] * phase_1_duration).T)
    # phase_1.addConstraint(cnsrt_1)

    timeline_1.registerPhase(phase_1)


    phase_2_duration = 3
    phase_2 = pyphase.Phase(phase_2_duration, 'phase_2')
    # phase_2.addItem(cnsrt_1)
    # phase_2.addParameterValues(param_1, np.array([[1, 1, 1]] * phase_1_duration).T, nodes=[1, 2])
    # phase_2.addCost(cost_1, nodes=[1, 2])
    phase_2.addVariableBounds(var_1, np.array([[1, 1, 1]] * phase_1_duration).T, np.array([[1, 1, 1]] * phase_1_duration).T, nodes=[1, 2])
    # phase_2.addConstraint(cnsrt_1)

    timeline_1.registerPhase(phase_2)
    print('=============== running ===============')

    timeline_1.addPhase(phase_1)
    timeline_1.addPhase(phase_2)

    # for elem in timeline_1.getActivePhases():
    #     print(elem.getName())



    print('=============== results ===============')
    # print("item1: ", cnsrt_1.getNodes())
    # ====================
    # print("param_1: \n", param_1.getValues())
    # ====================
    # print('cost_1: ', cost_1.getNodes())
    # ====================
    print('var_1:')
    print(var_1.getLowerBounds())
    print(var_1.getUpperBounds())

    print("=========> shifting once")
    tic = time.time()
    pm._shift_phases()
    pm._shift_phases()
    pm._shift_phases()
    pm._shift_phases()

    print('=============== results ===============')
    # print("item1: ", cnsrt_1.getNodes())
    # ====================
    # print("param_1: \n", param_1.getValues())
    # ====================
    # print('cost_1: ', cost_1.getNodes())
    # ====================
    print('var_1:')
    print(var_1.getLowerBounds())
    print(var_1.getUpperBounds())

    exit()