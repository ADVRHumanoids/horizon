from horizon.problem import Problem
import pprint
import numpy as np
import logging
import casadi as cs

nodes = 10
dt = 0.01
prb = Problem(nodes)
x = prb.createStateVariable('x', 6)
y = prb.createStateVariable('y', 3)
z = x[2:5]
print(type(z))
print(z)

print(type(z[2]))
print(z[2])

print(z + y)


