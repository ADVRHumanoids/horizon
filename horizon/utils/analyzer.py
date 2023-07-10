from horizon.problem import Problem
from horizon.functions import Function, Constraint
import numpy as np
import json, codecs


class bcolors:
    CPURPLE = '\033[95m'
    CBLUE0 = '\033[94m'
    CCYAN0 = '\033[96m'
    CGREEN0 = '\033[92m'
    CYELLOW0 = '\033[93m'
    CRED0 = '\033[91m'

    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CUNDERLINE = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'


class ProblemAnalyzer:
    def __init__(self, prb: Problem):
        self.prb = prb
        np.set_printoptions(suppress=False,  # suppress small results
                            linewidth=np.inf,
                            precision=3,
                            threshold=np.inf,  # number of displayed elements for array
                            formatter=None
                            )

        self._fun_color_dict()

    def _fun_color_dict(self):

        self._item_fun = {
            'elements': self.printElement,
            'nodes': self.printNodes,
            'values': self.printValues,
            'bounds': self.printBounds,
            'initial_guess': self.printInitialGuess
        }

        self._item_colors = {
            'elements': bcolors.CYELLOW0,
            'nodes': bcolors.CRED2,
            'values': bcolors.CVIOLET2,
            'bounds': bcolors.CBLUE,
            'initial_guess': bcolors.CCYAN0
        }

    def _separator(self):
        print(f"{bcolors.CBOLD}{bcolors.CWHITE}"
              f"-------------------------------------------------------------------------------------------------------"
              f"{bcolors.CEND}")

    def _print_title(self, title, color):
        print(f"{bcolors.CBOLD}{bcolors.CITALIC}{color} ================= {title} ==================== {bcolors.CEND}")

    def _print_lower_bounds(self, elem, color):
        print(f"{bcolors.CBOLD}{color}Lower Bounds:\n{bcolors.CEND}{elem.getLowerBounds()}")

    def _print_upper_bounds(self, elem, color):
        print(f"{bcolors.CBOLD}{color}Upper Bounds:\n{bcolors.CEND}{elem.getUpperBounds()}")

    def printBounds(self, elem, color):
        self._print_lower_bounds(elem, color)
        self._print_upper_bounds(elem, color)

    def printNodes(self, elem, color):
        print(f"{bcolors.CBOLD}{color}Nodes:\n{bcolors.CEND}{elem.getNodes()}")

    def printElement(self, elem, color):
        print(f"{bcolors.CBOLD}{color}Element:\n{bcolors.CEND}"
              f"{bcolors.CBOLD}{elem.getName()}{bcolors.CEND}"
              f" = {elem}")

    def printValues(self, elem, color):
        print(f"{bcolors.CBOLD}{color}Values:\n{bcolors.CEND}{elem.getValues()}")

    def printInitialGuess(self, elem, color):
        print(f"{bcolors.CBOLD}{color}Initial Guess:\n{bcolors.CEND}{elem.getInitialGuess()}")

    def printVariables(self, elem=None, suppress_nodes=False, suppress_bounds=False, suppress_ig=False):

        item_list = ['elements', 'nodes', 'bounds', 'initial_guess']

        if suppress_nodes is True:
            item_list.remove('nodes')

        if suppress_bounds is True:
            item_list.remove('bounds')

        if suppress_ig is True:
            item_list.remove('initial_guess')

        if elem is None:
            self._print_generic_group("VARIABLES", bcolors.CGREEN2, self.prb.getVariables(), item_list)
        else:
            self._print_generic_element(self.prb.getVariables(elem), item_list)

        # self._print_title("VARIABLES", bcolors.CGREEN2)
        # for name, elem in self.prb.getVariables().items():
        #     self._separator()
        #     self.printElement(elem, bcolors.CYELLOW0)
        #     self.printNodes(elem, bcolors.CRED2)
        #     self.printBounds(elem, bcolors.CBLUE)
        #     self.printInitialGuess(elem, bcolors.CCYAN0)

    def printParameters(self, elem=None, suppress_values=False):

        item_list = ['elements', 'nodes', 'values']

        if suppress_values is True:
            item_list.remove('values')

        if elem is None:
            self._print_generic_group("PARAMETERS", bcolors.CGREEN2, self.prb.getParameters(), item_list)
        else:
            self._print_generic_element(self.prb.getParameters(elem), item_list)
        # self._print_title("PARAMETERS", bcolors.CGREEN2)
        # for name, elem in self.prb.getParameters().items():
        #     self.printElement(elem, bcolors.CYELLOW0)
        #     self.printNodes(elem, bcolors.CRED2)
        #     self.printValues(elem, bcolors.CVIOLET2)

    def printConstraints(self, elem: str = None, suppress_bounds=False):

        item_list = ['elements', 'nodes', 'bounds']

        if suppress_bounds is True:
            item_list.remove('bounds')

        if elem is None:
            self._print_generic_group("CONSTRAINTS", bcolors.CGREEN2, self.prb.getConstraints(), item_list)
        else:
            self._print_generic_element(self.prb.getConstraints(elem), item_list)
        # self._print_title("CONSTRAINTS", bcolors.CGREEN2)
        # for name, elem in self.prb.getConstraints().items():
        #     self.printElement(elem, bcolors.CYELLOW0)
        #     self.printNodes(elem, bcolors.CRED2)
        #     self.printBounds(elem, bcolors.CBLUE)

    def printCosts(self, elem=None):

        item_list = ['elements', 'nodes']

        if elem is None:
            self._print_generic_group("COSTS", bcolors.CGREEN2, self.prb.getCosts(), item_list)
        else:
            self._print_generic_element(self.prb.getCosts(elem), item_list)
        # self._print_title("COSTS", bcolors.CGREEN2)
        # for name, elem in self.prb.getCosts().items():
        #     self.printElement(elem, bcolors.CYELLOW0)
        #     self.printNodes(elem, bcolors.CRED2)

    def _print_generic_group(self, name, color, container, item_list):
        self._print_title(name, color)
        for name, elem in container.items():
            self._print_generic_element(elem, item_list)
            self._separator()

    def _print_generic_element(self, elem, item_list):
        for item in item_list:
            self._item_fun[item](elem, self._item_colors[item])

    def print(self):
        self.printVariables()
        self.printParameters()
        self.printConstraints()
        self.printCosts()

    def prbAsDict(self):
        data = dict()

        data['n_nodes'] = self.prb.getNNodes() - 1

        # save state variables
        data['state'] = list()
        for sv in self.prb.getState():
            var_data = dict()
            var_data['name'] = sv.getName()
            var_data['size'] = sv.size1()
            var_data['lb'] = sv.getLowerBounds().flatten('F').tolist()
            var_data['ub'] = sv.getUpperBounds().flatten('F').tolist()
            var_data['initial_guess'] = sv.getInitialGuess().flatten('F').tolist()
            data['state'].append(var_data)

        # save input variables
        data['input'] = list()
        for sv in self.prb.getInput():
            var_data = dict()
            var_data['name'] = sv.getName()
            var_data['size'] = sv.size1()
            var_data['lb'] = sv.getLowerBounds().flatten('F').tolist()
            var_data['ub'] = sv.getUpperBounds().flatten('F').tolist()
            var_data['initial_guess'] = sv.getInitialGuess().flatten('F').tolist()
            data['input'].append(var_data)

        # save parameters
        data['param'] = dict()
        for p in self.prb.getParameters().values():
            var_data = dict()
            var_data['name'] = p.getName()
            var_data['size'] = p.getDim()
            var_data['values'] = p.getValues().flatten('F').tolist()
            data['param'][var_data['name']] = var_data

        # save cost and constraints
        data['cost'] = dict()
        for f in self.prb.getCosts().values():
            f: Function = f
            var_data = dict()
            var_data['name'] = f.getName()
            var_data['repr'] = str(f.getFunction())
            var_data['var_depends'] = [v.getName() for v in f.getVariables()]
            var_data['param_depends'] = [v.getName() for v in f.getParameters()]
            var_data['nodes'] = f.getNodes() if isinstance(f.getNodes(), list) else f.getNodes().tolist()
            var_data['function'] = f.getFunction().serialize()
            data['cost'][var_data['name']] = var_data

        data['constraint'] = dict()
        for f in self.prb.getConstraints().values():
            f: Function = f
            var_data = dict()
            var_data['name'] = f.getName()
            var_data['repr'] = str(f.getFunction())
            var_data['var_depends'] = [v.getName() for v in f.getVariables()]
            var_data['param_depends'] = [v.getName() for v in f.getParameters()]
            var_data['nodes'] = f.getNodes() if isinstance(f.getNodes(), list) else f.getNodes().tolist()
            var_data['function'] = f.getFunction().serialize()
            var_data['lb'] = f.getLowerBounds().flatten('F').tolist()
            var_data['ub'] = f.getUpperBounds().flatten('F').tolist()
            data['constraint'][var_data['name']] = var_data

        return data
    # @staticmethod
    # def compare(data1, data2):
    #
    #     print("comparing d1 to d2:")
    #     self.findDiff(data1, data2)


def compare_dicts(dict1, dict2, keys=[]):
    for k in dict1.keys() | dict2.keys():
        new_keys = keys + [k]
        if isinstance(dict1.get(k), dict) and isinstance(dict2.get(k), dict):
            compare_dicts(dict1[k], dict2[k], new_keys)
        else:
            if dict1.get(k) != dict2.get(k):
                colored_keys = [f"{bcolors.CYELLOW0}{bcolors.CBOLD if i == 0 else ''}{k}" for i, k in
                                enumerate(new_keys)]
                print(f"{bcolors.CBOLD}{'.'.join(colored_keys)}{bcolors.CEND}")

                if isinstance(dict1.get(k), list) and isinstance(dict2.get(k), list):
                    if len(dict1.get(k)) != len(dict2.get(k)):
                        print(
                            " " * 3 + f"{bcolors.CRED2}Difference of size: {len(dict1.get(k))} != {len(dict2.get(k))}{bcolors.CEND}")
                    else:
                        diffs = [i for i, (elem1, elem2) in enumerate(zip(dict1.get(k), dict2.get(k))) if
                                 elem1 != elem2]
                        print(
                            " " * 3 + f"{bcolors.CRED2}Difference at position(s): {diffs}{bcolors.CEND}") if diffs else None
                else:
                    print(f"Element is not a list")
                print(" " * 6 + f"{bcolors.CCYAN0}- : {dict1.get(k)}{bcolors.CEND}")
                print(" " * 6 + f"{bcolors.CVIOLET2}+ : {dict2.get(k)}{bcolors.CEND}")


if __name__ == '__main__':
    obj_text_1 = codecs.open('data.json', 'r', encoding='utf-8').read()
    obj_text_2 = codecs.open('data_add_nodes.json', 'r', encoding='utf-8').read()
    data1 = json.loads(obj_text_1)
    data2 = json.loads(obj_text_2)

    print("comparing d1 to d2:")
    compare_dicts(data1, data2)
