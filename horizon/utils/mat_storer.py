from scipy.io import savemat, loadmat
import numpy as np
import argparse

import h5py

"""
setInitialGuess assign initial guesses given a solution guess
Args:
    variables: a dictionary containing the variables. The keys in the dict should be the same used to store a solution
    solution: a solution dict
"""
def setInitialGuess(variables: dict, solution: dict):
    for variable_name, variable in variables.items():
        if variable_name in solution.keys():
            if len(variable.getNodes()) == 1: #this is to handle a special case, e.g. final time optimization
                variable.setInitialGuess(solution[variable_name].T)
            else:
                node = 0
                for guess in solution[variable_name].T:
                    variable.setInitialGuess(guess, node)
                    node += 1
        else:
            print(f"variable {variable_name} not in solution dict, skipped")

class matStorerIO:
    def __init__(self):
        self.__parser = argparse.ArgumentParser(description="Save and Load solutions")
        self.__parser.add_argument("--save", help="enable save of solution", type=str)
        self.__parser.add_argument("--load", help="enable load of solution", type=str)

        self.dict_values = dict()
        self.argParse()

    def argParse(self):
        self.__args = self.__parser.parse_args()

    def load(self, solution: dict):
        if self.__args.load:
            file_name = self.__args.load
            print(f"loading file {file_name}")
            solution.update(loadmat(file_name))
            return True
        return False

    def append(self, dict_values):
        self.dict_values.update(dict_values)

    def save(self):
        if self.__args.save:
            file_name = self.__args.save
            print(f"saving file {file_name}")
            savemat(file_name, self.dict_values)
            return True
        return False

    def store(self, dict_values):
        self.append(dict_values)
        return self.save()

class matStorer:
    def __init__(self, file_name):
        self.file_name = file_name
        self.dict_values = dict()

    # def append(self, values):
    #     with open(self.file_name, 'ab') as f:
    #         savemat(f, {'pad': values})

    def append(self, dict_values):
        self.dict_values.update(dict_values)

    def store(self, dict_values):
        self.append(dict_values)
        savemat(self.file_name, self.dict_values)  # write

    def save(self):
        savemat(self.file_name, self.dict_values)

    def load(self):
        return loadmat(self.file_name)

if __name__ == '__main__':
    a = np.ones([1,5])
    b = 2* np.ones([1,5])
    filename = 'try.mat'
    ms = matStorer(filename)
    ms.store({'a': a})
    mat = ms.load()
    print(mat)
    # mat = loadmat(filename)
    # print(mat)


