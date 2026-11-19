"""Python setup for native iLQR/IPOPT graphs; numerical evaluation stays in C++.

Native calls return MX placeholders. Finalize their expressions through the
same manager before constructing a solver; ordinary CasADi Functions cannot
evaluate these placeholders. Serialization is used only during graph setup.
"""
import itertools

import casadi as cs
from horizon.solvers import pyilqr

Function = pyilqr.Function
CasadiFunction = pyilqr.CasadiFunction
StateSpace = pyilqr.StateSpace
EuclideanSpace = pyilqr.EuclideanSpace


class FunctionManager(pyilqr.FunctionManager):
    """Registry of native or CasADi maps receiving [state; control; parameters]."""

    def __init__(self):
        super().__init__()
        self._signatures = {}
        self._casadi_callbacks = []

    def add(self, name, function, parameters=()):
        parameters = [p if isinstance(p, pyilqr.Parameter) else pyilqr.Parameter(*p)
                      for p in parameters]
        super().add(name, function, parameters)
        self._signatures[name] = parameters

    def addCasadi(self, name, function, exact_hessian=True):
        self.add(name, CasadiFunction(function, exact_hessian),
                 [(function.name_in(i), function.size1_in(i))
                  for i in range(2, function.n_in())])

    def asCasadi(self, name, nx, nu):
        """Return an MX Function with native values and analytical Jacobians.

        The returned Function retains its callbacks through this manager; keep
        the manager alive as long as the Function. Each adapter owns a clone of
        the native graph. Second derivatives are intentionally unavailable;
        use a Gauss--Newton NLP Hessian for native residuals/constraints.
        """
        parameters = self._signatures[name]
        sizes = [nx, nu] + [p.size for p in parameters]
        callback = cs.Function()
        owner = self._exportCasadi(name, callback)
        self._casadi_callbacks.append(owner)
        if sum(sizes) != callback.size1_in(0):
            raise ValueError("State/control sizes do not match the registered function")
        inputs = [cs.MX.sym(n, size) for n, size in
                  zip(["x", "u"] + [p.name for p in parameters], sizes)]
        return cs.Function(name, inputs, [callback(cs.vertcat(*inputs))],
                           ["x", "u"] + [p.name for p in parameters], ["value"])


class SymbolicFunctionManager(FunctionManager):
    """Compose MX expressions around explicitly registered native functions."""

    _ids = itertools.count()

    def __init__(self):
        super().__init__()
        self._calls = []

    def addNative(self, name, function):
        self.add(name, function)

        def call(argument):
            argument = cs.MX(argument)
            if argument.shape != (function.inputSize(), 1):
                raise ValueError("Native call input must be a column with inputSize() entries")
            result = cs.MX.sym(f"native_{name}_{next(self._ids)}", function.outputSize())
            self._calls.append((argument, result, function))
            return result

        return call

    def addExpression(self, name, inputs, expression, exact_hessian=True):
        inputs = list(inputs)
        graph = [*inputs, cs.MX(expression)]
        for argument, result, _ in self._calls:
            graph.extend((argument, result))
        # Problem states/controls may concatenate several symbolic variables.
        # Lower them to the dense stage symbols required by the native builder,
        # consistently replacing symbols inside nested native-call arguments.
        if any(not cs.MX(value).is_symbolic() for value in inputs[:2]):
            packed = [cs.MX.sym("x", inputs[0].numel()),
                      cs.MX.sym("u", inputs[1].numel())]
            graph = [*packed, *inputs[2:],
                     *cs.substitute(graph[len(inputs):], inputs[:2], packed)]
            inputs = [*packed, *inputs[2:]]
        serializer = cs.StringSerializer()
        serializer.pack(graph)
        self._addExpression(name, serializer.encode(), len(inputs),
                            [call[2] for call in self._calls], exact_hessian)
        self._signatures[name] = [pyilqr.Parameter(p.name(), p.numel()) for p in inputs[2:]]

    def addResidual(self, name, inputs, expression):
        self.addExpression(name, inputs, expression, exact_hessian=False)

    def addCost(self, name, inputs, expression):
        self.addExpression(name, inputs, expression, exact_hessian=True)
