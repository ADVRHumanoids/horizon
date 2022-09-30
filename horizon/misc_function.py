import numpy as np
import warnings

def unravelElements(elements):
    if isinstance(elements, int):
        unraveled_elem = [elements]
        pass
    elif any(isinstance(el, list) for el in elements):
        unraveled_elem = list()
        for el in elements:
            temp = list(range(el[0], el[1]+1)) # +1 # todo cannot add+1 here?
            for item in temp:
                unraveled_elem.append(item) if item not in unraveled_elem else unraveled_elem
    elif isinstance(elements, list):
        unraveled_elem = list()
        temp = list(range(elements[0], elements[1]+1)) # +1
        for item in temp:
            unraveled_elem.append(item)

    return unraveled_elem


def listOfListFLOATtoINT(listOfList):
    # transform every element to INT
    for i in range(len(listOfList)):
        if isinstance(listOfList[i], list):
            for j in range(len(listOfList[i])):
                listOfList[i][j] = int(listOfList[i][j])
        else:
            listOfList[i] = int(listOfList[i])

    return listOfList

def checkNodes(nodes, nodes_array=None):

    nodes_vec = np.array(nodes).astype(int)
    # todo check for repeated nodes
    if nodes_array is None:
        checked_nodes = nodes_vec
        discarded_nodes = np.array([])
    else:
        # get from nodes only the nodes active in nodes_array
        # example: nodes_array = [0 0 1 0 1 1 0 0]
        #                nodes = [2, 3, 4]
        #       1. get from nodes array only the elements at position [2, 3, 4]  --> [1 0 1]
        #       2. mask 'nodes' with [1 0 1] --> [2, 4]
        checked_nodes = np.ma.masked_array(nodes_vec, mask=np.logical_not(nodes_array[nodes_vec])).compressed()
        discarded_nodes = np.ma.masked_array(nodes_vec, mask=nodes_array[nodes_vec]).compressed()

        # todo: this sucks, as it does not tells you which item calls this
        if checked_nodes.size != nodes_vec.size:
            warnings.warn(f'Element requested is not defined/active on node: {discarded_nodes}.')

    return checked_nodes, discarded_nodes

def checkValueEntry(val):
    if isinstance(val, (int, float)):
        val = np.array([val])
    else:
        val = np.array(val)

        # if single value, flatten
        # note: dont flatten matrix of values!
        multiple_vals = val.ndim == 2 and val.shape[1] != 1

        if not multiple_vals:
            # transform everything into a (n x 1) matrix (columns --> nodes, rows --> dim)
            val = np.reshape(val, (val.size, 1))

    return val

def convertNodestoPos(nodes, nodes_array):
    # todo add guards
    nodes_to_pos = np.nonzero(np.in1d(np.where(nodes_array == 1), nodes))[0]
    return nodes_to_pos

def getBinaryFromNodes(total_nodes: int, active_nodes: list):
    # add guards
    # if not isinstance(nodes, list):
    #     raise TypeError('input must be a list of nodes')

    nodes_array = np.zeros(total_nodes).astype(int)
    nodes_array[active_nodes] = 1

    return nodes_array

def getNodesFromBinary(nodes_array):
    # add guards
    # if not isinstance(nodes_array, list):
    #     raise TypeError('input must be a list of binary nodes')
    nodes = np.where(nodes_array == 1)[0]

    return nodes

def shift_array(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:, :num] = fill_value
        result[:, num:] = arr[:, :-num]
    elif num < 0:
        result[:, num:] = fill_value
        result[:, :num] = arr[:, -num:]
    else:
        result[:] = arr
    return result

if __name__ == '__main__':

    print(checkNodes([3,3,4,5,6], np.array([0, 0, 1, 1, 1, 0, 0, 1, 0])))