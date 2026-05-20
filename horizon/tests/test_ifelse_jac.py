import casadi as cs 

x = cs.SX.sym('x', 1)

y = cs.if_else(cs.fabs(x) > 0.1, 1/x, x)

jac = cs.Function('jj', [x], [cs.jacobian(y, x)])
print(jac)
print(jac(x))
print(jac(0))
