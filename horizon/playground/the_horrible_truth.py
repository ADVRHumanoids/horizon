#!/usr/bin/env python3

import horizon.problem as prb
import horizon.utils.plotter as plotter
import casadi as cs
import numpy as np
from horizon.transcriptions.transcriptor import Transcriptor
from horizon.solvers import solver
import matplotlib.pyplot as plt

n_nodes = 10
dt = 0.1
mu = 0.2
grav = 9.81
prob = prb.Problem(n_nodes, receding=True, casadi_type=cs.SX) # receding=True,

p = prob.createStateVariable('pos', dim=2)
v = prob.createStateVariable('vel', dim=2)
F = prob.createInputVariable('force', dim=2)


print('========= constructing constraint in nodes: [3, 4, 5, 8, 9] =============')
cnsrt = prob.createIntermediateConstraint('some', p + v, nodes=[3, 4, 5, 8, 9])
print(cnsrt.getLowerBounds())

enabled_node = [4]
print(f'========= disabling all but node: {enabled_node} =============')
cnsrt.setNodes(enabled_node)
print(cnsrt.getLowerBounds())

print(f'========= enable all nodes =============')
cnsrt.setNodes(range(n_nodes+1))
print(cnsrt.getLowerBounds())

print('========= setting bounds to nodes: [3, 8, 9] =============')
cnsrt.setLowerBounds([2, 2], nodes=[3, 8, 9])
lb_1 =cnsrt.getLowerBounds().copy()
print(lb_1)

print('========== disabling all nodes ============')
cnsrt.setNodes([])
print(cnsrt.getLowerBounds())

print('========== setting bounds to disabled nodes: should throw warning ============')
cnsrt.setLowerBounds([2, 2], nodes=[6, 7, 8])
print(cnsrt.getLowerBounds())

print('=========== enabling all nodes (if not receding, should throw) ===========')
print(f'=========== should be: {lb_1} ===========')
cnsrt.setNodes(range(n_nodes+1))
print(cnsrt.getLowerBounds())

print('========== enabling nodes: [2, 3, 4] ============')
cnsrt.setNodes([2, 3, 4])
print(cnsrt.getLowerBounds())

print('========== enabling all nodes ============')
cnsrt.setNodes(range(n_nodes+1))
print(cnsrt.getLowerBounds())

node_to_set = 7
print(f'========== setting bound to node: {node_to_set} ============')
cnsrt.setLowerBounds([10, 10], nodes=node_to_set)
print(cnsrt.getLowerBounds())

node_list = [node for node in range(10) if node != node_to_set]
print(f'========== enable nodes: {node_list} ============')
cnsrt.setNodes(node_list)
print(cnsrt.getLowerBounds())
print('========== re-enable last set node ============')
cnsrt.setNodes(range(11))
print(cnsrt.getLowerBounds())





