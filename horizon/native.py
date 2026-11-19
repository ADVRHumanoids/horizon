"""Native numerical functions embedded in ordinary CasADi MX expressions.

Problem and Solver manage modeling and compilation. This module contains no
second problem builder and creates no FunctionManager during expression setup.
"""
import itertools
import weakref

import casadi as cs
import numpy as np

_primitives = weakref.WeakValueDictionary()
_ids = itertools.count()


def _outputs(expression):
    seen = set()
    result = []
    def visit(node):
        key = hash(node)
        if key in seen:
            return
        seen.add(key)
        for i in range(node.n_dep()):
            visit(node.dep(i))
        if node.is_output() and node.dep(0).is_call():
            result.append(node)
    visit(cs.MX(expression))
    return result


def _expanded(expression):
    expression = cs.MX(expression)
    for _ in range(100):
        old, new = [], []
        for output in _outputs(expression):
            call = output.dep(0)
            function = call.which_function()
            if function.name().startswith("horizon_native_"):
                continue
            if function.is_a("MXFunction") or function.is_a("SXFunction"):
                arguments = [call.dep(i) for i in range(call.n_dep())]
                if function.is_a("MXFunction"):
                    bodies = [None]*function.n_out()
                    for i in range(function.n_instructions()):
                        instruction = function.instruction_MX(i)
                        if instruction.op() == cs.OP_OUTPUT:
                            bodies[function.instruction_output(i)[0]] = instruction.dep(0)
                    values = cs.substitute(bodies, function.mx_in(), arguments)
                else:
                    inline = function.expand(function.name()+"_inline", {"never_inline": False})
                    values = inline.call(arguments, True, False)
                old.append(output)
                new.append(values[output.which_output()])
        if not old:
            return expression
        expression = cs.graph_substitute(expression, old, new)
    raise ValueError("Native graph has excessive function nesting")


def _calls(expression):
    if not isinstance(expression, cs.MX):
        return []
    result = []
    for output in _outputs(_expanded(expression)):
        call = output.dep(0)
        name = call.which_function().name()
        if name.startswith("horizon_native_"):
            primitive = _primitives.get(name)
            if primitive is None:
                raise RuntimeError("Keep the NativeRobot/NativeFunction alive until the expression is registered with Problem")
            result.append((call.dep(0), output, primitive))
    return result


def _owners(expression):
    return tuple(dict.fromkeys(call[2] for call in _calls(expression)))


class NativeFunction:
    """A native Function callable on numeric vectors or symbolic MX arguments.

    Retain this object while building standalone expressions. Problem retains
    its owners once an expression is registered as a cost, constraint or dynamics.
    """
    def __init__(self, function, *, configuration_space=None, nq=None):
        self.native = function
        self.configuration_space = configuration_space
        self.nq = nq
        self.name = f"horizon_native_{next(_ids)}"
        self._function = cs.Function()
        self._owner = function._exportCasadi(self.name, self._function)
        _primitives[self.name] = self

    def __call__(self, *arguments):
        if any(isinstance(a, cs.SX) for a in arguments):
            raise TypeError("Native robot expressions require MX variables")
        if any(isinstance(a, cs.MX) for a in arguments):
            argument = cs.vertcat(*arguments)
            if argument.shape != (self.native.inputSize(), 1):
                raise ValueError("Native function input dimension mismatch")
            return self._function(argument)
        argument = np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in arguments])
        return self.native.evaluate(argument)
