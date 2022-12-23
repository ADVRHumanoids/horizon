import numpy as np
import warnings
import time
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

def index_finder(lst, item):
    """A generator function, if you might not need all the indices"""
    start = 0
    while True:
        try:
            start = lst.index(item, start)
            yield start
            start += 1
        except ValueError:
            break

def index_find_all(lst, item, results=None):
    """ If you want all the indices.
    Pass results=[] if you explicitly need a list,
    or anything that can .append(..)
    """
    if results is None:
        length = len(lst)
        results = (array.array('B') if length <= 2**8 else
                   array.array('H') if length <= 2**16 else
                   array.array('L') if length <= 2**32 else
                   array.array('Q'))
    start = 0
    while True:
        try:
            start = lst.index(item, start)
            results.append(start)
            start += 1
        except ValueError:
            return results

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

    # nodes_vec = np.array(nodes).astype(int)
    nodes_vec = nodes if hasattr(nodes, '__iter__') else [nodes]
    # todo check for repeated nodes
    if nodes_array is None:
        checked_nodes = nodes_vec
    else:
        # get from nodes only the nodes active in nodes_array
        checked_nodes = [node for node in nodes_vec if nodes_array[node] == 1]
        # slower:
        # checked_nodes = [node for node in nodes_vec if nodes_array[node] == 1]
        # discarded_nodes = [node for node in nodes_vec if nodes_array[node] == 0]
        # slowest:
        # checked_nodes = np.ma.masked_array(nodes_vec, mask=np.logical_not(nodes_array[nodes_vec])).compressed()
        # discarded_nodes = np.ma.masked_array(nodes_vec, mask=nodes_array[nodes_vec]).compressed()


        # todo: this sucks, as it does not tells you which item calls this ALSO SLOW STUFF DOWN
        # if checked_nodes.size != nodes_vec.size:
            # warnings.warn(f'Element requested is not defined/active on node: {discarded_nodes}.')

    # return np.array(checked_nodes), np.array(discarded_nodes)
    return checked_nodes

def checkValueEntry(val):
    # todo: if the array is monodimensional, it should be considered as column or row?
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

def convertNodestoPosNumpy(nodes, nodes_array):
    # todo add guards
    nodes_to_pos = np.nonzero(np.in1d(np.where(nodes_array == 1), nodes))[0]

    return nodes_to_pos

# todo this is probably slowing down everything. Better to have a complete array always with all the nodes instead of having only
# the active ones
def convertNodestoPos(nodes, nodes_array):
    # todo add guards
    # nodes_to_pos = np.nonzero(np.in1d(np.where(nodes_array == 1), nodes))[0]
    nodes_to_pos = []
    index = 0
    for i in range(len(nodes_array)):
        val = nodes_array[i]
        if val == 1 and i in nodes:
            nodes_to_pos.append(index)
        index += val

    return nodes_to_pos
    # start = nodes[0]
    # for i in range(len(nodes)):
    #     try:
    #         start = nodes_array.index(1, start)
    #         yield start
    #         start += 1
    #     except ValueError:
    #         break

def getBinaryFromNodes(total_nodes: int, active_nodes: list):
    # add guards
    # if not isinstance(nodes, list):
    #     raise TypeError('input must be a list of nodes')

    nodes_array = np.zeros(total_nodes).astype(int)
    nodes_array[active_nodes] = 1

    return nodes_array

# todo also this should be used as little as possible
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

    # tic = time.time()
    # print(checkNodes([3, 4, 7], np.array([0, 0, 1, 1, 1, 0, 0, 1, 0])))
    # print(time.time() - tic)

    tic = time.time()
    print(convertNodestoPosNumpy([2, 4], np.array([0, 1, 0, 1, 1])))
    print(time.time() - tic)
    tic = time.time()
    print(*convertNodestoPos([2, 3, 8], [0, 1, 0, 1, 1]))
    print(time.time() - tic)


