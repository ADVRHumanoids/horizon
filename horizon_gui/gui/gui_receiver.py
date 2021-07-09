from horizon import problem as horizon
import parser
import re
import casadi as cs
import math

class horizonImpl():
    def __init__(self, nodes, logger=None):

        # #todo logger! use it everywhere!
        self.logger = logger

        self.nodes = nodes
        self.casadi_prb = horizon.Problem(self.nodes)

        self.sv_dict = dict()  # state variables
        self.fun_dict = dict() # functions

        # self.active_fun_list = list()

    def addStateVariable(self, data):

        name = data['name']
        dim = data['dim']
        prev = data['prev']

        flag, signal = self.checkStateVariable(name)
        if flag:
            if prev == 0:
                var = self.casadi_prb.createStateVariable(name, dim)
            else:
                var = self.casadi_prb.createStateVariable(name, dim, prev)

            self.sv_dict[name] = dict(var=var, dim=dim)

            return True, signal
        else:
            return False, signal

    def addFunction(self, data):

        # get Function from Text
        name = data['name']
        str_fun = data['str']

        flag, signal = self.checkFunction(name, str_fun)
        if flag:
            flag_syntax = self._createAndAppendFun(name, str_fun)
            if flag_syntax:
                return True, signal
            else:
                return False, "Syntax is wrong."
        else:
            return False, signal

    def activateFunction(self, name, fun_type):

        flag, signal = self.checkActiveFunction(name)

        if flag:
            if fun_type == 'constraint':
                try:
                    # self.logger.info('Adding function: {}'.format(self.fun_dict[name]))
                    active_fun = self.casadi_prb.createConstraint(name, self.fun_dict[name]['fun'])
                    # self.active_fun_list.append(active_fun)
                    self.fun_dict[name].update({'active': active_fun})
                except Exception as e:
                    return False, e

            elif fun_type == 'costfunction':
                try:
                    active_fun = self.casadi_prb.createCostFunction(name, self.fun_dict[name]['fun'])
                    # self.active_fun_list.append(active_fun)
                    self.fun_dict[name].update({'active': active_fun})
                except Exception as e:
                    return False, e

            return True, signal + 'Function "{}" activated as "{}".'.format(name, active_fun.getType())
        else:
            return False, signal

    def removeActiveFunction(self, name):

        active_fun_type = self.fun_dict[name]['active'].getType()
        self.fun_dict[name]['active'] = None

        if active_fun_type == 'constraint':
            self.casadi_prb.removeConstraint(name)
        elif active_fun_type == 'costfunction':
            self.casadi_prb.removeCostFunction(name)
        else:
            return False, 'Function type "{}" not recognized'.format(active_fun_type)

        return True, 'Function "{}" successfully removed.'.format(name)

    def removeStateVariable(self, data):
        print('"removeStateVariable" yet to implement. Data: {}'.format(data))

    def solveProblem(self):
        print('"solveProblem" yet to implement')

    def checkActiveFunction(self, name):

        if self.fun_dict[name]['active'] is not None:
            signal = "active function already inserted"
            return False, signal

        return True, 'Function "{}" can be activated. Adding.'.format(name)

    def checkFunction(self, name, fun): # fun, name

        if name in self.fun_dict.keys():
            signal = "function already Inserted"
            return False, signal

        elif name == "":
            signal = "Empty Name of Function Not Allowed"
            return False, signal

        if fun == "":
            signal = "Empty Function Not Allowed"
            return False, signal

        return True, 'Function "{}" is acceptable. Adding..'.format(name)

    def checkStateVariable(self, name):

        if name == "":
            signal = "State Variable: Empty Value Not Allowed"
            return False, signal
        elif name in self.sv_dict.keys():
            signal = "State Variable: Already Inserted"
            return False, signal
        elif name.isnumeric():
            signal = "State Variable: Invalid Name"
            return False, signal

        return True, "State Variable: generated '{}'".format(name)


    def fromTxtToFun(self, str_fun):

        fun = None
        # todo CHECK IF STR_FUN IS CORRECT?

        # todo better approach? the problem is that i should have here a set of variables
        # i don't want to write on the GUI self.x or worse self.horizon_receiver.sv_dict[]... better ideas?
        # is it possible for some of the variables not to be substituted?

        # get from text all variables and substitute them with self.horizon_receiver.sv_dict[''] ..

        # todo add also generic functions
        dict_vars = dict()
        for var in self.sv_dict.keys():
            dict_vars[var] = "self.sv_dict['{}']['var']".format(var)

        for var in self.fun_dict.keys():
            dict_vars[var] = "self.fun_dict['{}']['fun']".format(var)

        # these are all the state variable found in sv_dict and fun_dict
        all_variables = list(self.sv_dict.keys()) + list(self.fun_dict.keys())

        regex_vars = '\\b|'.join(sorted(re.escape(k) for k in all_variables))
        regex_math = '\\b|'.join(sorted(re.escape(k) for k in self.getValidOperators()['math']))

        # If repl is a function, it is called for every non-overlapping occurrence of pattern.
        modified_fun = re.sub(regex_vars, lambda m: dict_vars.get(m.group(0)), str_fun, flags=re.IGNORECASE)
        modified_fun = re.sub(regex_math, lambda m: 'math.{}'.format(m.group(0)), modified_fun, flags=re.IGNORECASE)

        # parse str to code
        try:
            res = parser.expr(modified_fun)
        except Exception as e:
            self.logger.warning(e)
            return fun

        code = res.compile()


        # todo add try exception + logger

        try:
            fun = eval(code)
        except Exception as e:
            self.logger.warning(e)

        return fun

    def editFunction(self, name, str_fun):

        if name in self.fun_dict.keys():
            flag_syntax, signal_syntax = self._createAndAppendFun(name, str_fun)
            if flag_syntax:
                signal = 'Function "{}" edited with {}. Updated function: {}'.format(name, str_fun, self.fun_dict[name])
                return True, signal
            else:
                return False, signal_syntax
        else:
            signal = 'Failed editing of function "{}".'.format(name)
            return False, signal

    def updateFunctionNodes(self, name, nodes):
        # small hack to include also the max for each couple of min/max --> [0, 3] --> 0, 1, 2, 3. Not 0, 1, 2
        # for couples in nodes:
        #     couples[1] = couples[1]+1

        self.fun_dict[name]['active'].setNodes(nodes, erasing=True)

    def updateFunctionUpperBounds(self, name, ub, nodes):
        self.fun_dict[name]['active'].setUpperBounds(ub, nodes)

    def updateFunctionLowerBounds(self, name, ub, nodes):
        self.fun_dict[name]['active'].setLowerBounds(ub, nodes)

    def updateFunctionBounds(self, name, lb, ub, nodes):
        self.fun_dict[name]['active'].setBounds(lb, ub, nodes)

    def getFunctionDict(self):
        return self.fun_dict

    def getFunction(self, name):
        if name in self.fun_dict.keys():
            return self.fun_dict[name]
        #todo change? return only active?

    def getVarDict(self):
        return self.sv_dict

    def getVar(self, elem):

        return self.sv_dict[elem]

    def getNodes(self):
        return self.nodes

    def setHorizonNodes(self, n_nodes):
        self.nodes = n_nodes
        self.casadi_prb.setNNodes(self.nodes)

    def _createAndAppendFun(self, name, str_fun):

        fun = self.fromTxtToFun(str_fun)
        # TODO I can probably do a wrapper function in casadi self.createFunction(name, fun, type)
        # TODO HOW ABOUT GENERIC FUNCTION? Should i do a casadi function for them?
        # fill horizon_receiver.fun_dict and funList
        # TODO add fun type??
        if fun is not None:
            self.fun_dict[name] = dict(fun=fun, str=str_fun, active=None)
            return True
        else:
            return False

    def generateProblem(self):
        # try:
        self.casadi_prb.createProblem()
        # except Exception as e:
        #     return self.logger.warning(e)

    def solve(self):
        try:
            self.casadi_prb.solveProblem()
        except Exception as e:
            return self.logger.warning(e)

    def getInfoAtNodes(self, node):
        vars = list()
        vars_dict = self.casadi_prb.scopeNodeVars(node) #lb, ub, w0
        if vars_dict is not None:
            for var in vars_dict.values():
                vars.append(var)

        return vars

    def serialize(self):

        self.casadi_prb.serialize()

        # serialize state variables
        for name, data in self.sv_dict.items():
            self.logger.debug('Serializing variable "{}": {}'.format(name, data['var']))
            self.sv_dict[name]['var'] = data['var'].serialize()

        # serialize functions
        for name, data in self.fun_dict.items():
            self.logger.debug('Serializing function "{}": {}'.format(name, data['fun']))
            self.fun_dict[name]['fun'] = data['fun'].serialize()

    def deserialize(self):

        self.casadi_prb.deserialize()

        # deserialize state variables
        for name, data in self.sv_dict.items():
            self.sv_dict[name]['var'] = cs.SX.deserialize(data['var'])

        # deserialize functions
        for name, data in self.fun_dict.items():
            self.fun_dict[name]['fun'] = cs.SX.deserialize(data['fun'])

    def getValidOperators(self):
        '''
        return dictionary:
        keys: packages imported
        values: all the elements from the imported package that are considered "valid"
        '''

        full_list = dict()
        full_list['math'] = [elem for elem in dir(math) if not elem.startswith('_')]
        full_list['cs'] = ['cs.' + elem for elem in dir(cs) if not elem.startswith('_')]
        return full_list

if __name__ == '__main__':

    impl = horizonImpl(20)
    impl.addStateVariable(dict(name='x', dim=3, prev=0))


