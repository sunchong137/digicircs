import os
import copy
import itertools
import numpy
import random
import warnings
import string
from digicircs import __config__


def make_dir(path_name):
    '''
    Make directory if it does not exist.
    '''
    if not os.path.exists(path_name):
        os.makedirs(path_name)

def ravel_list(lst: list, dim: int = None, max_iter: int = 20):
    '''
    Transform a multi-dimension list to a 1D list.
    input:
        :param lst: the multi-dimension list to be raveled.
        :param dim: the dimension of the lst.
        :param max_iter: maximum iteration.
    returns:
        A 1D list.
    Example:
        [[[], [0,1]], [[1,2]]] -> [0,1,1,2]
    Note: lists with uncertain dimensions cannot be raveled here.
    '''
    if dim is not None:
        if dim > max_iter:
            raise Exception("Make sure the dimension of your list is smaller than %d!"%max_iter)
        else:
            max_iter = dim
    _lst = copy.copy(lst)
    for i in range(max_iter):
        try:
            _lst = list(itertools.chain.from_iterable(_lst))
        except:
            break

    return _lst

def get_num_gates_topo(topo_lst: list):
    '''
    Get the number of gates in a circuit topology represented by a list.
    input:
        :param topo_lst: the list storing the topology of circuits.
                         with structure[[[layer 1, 1q-gate-qubits],[layer 1, 2q-gate-qubits]],
                                        [[layer 2, 1q-gate-qubits],[layer 2, 2q-gate-qubits]],
                                        ...
                                        ]
    returns:
        the number of gates in this circuit.
    '''
    n_gates = 0
    for layer in topo_lst:
        n_gates += len(layer[0])
        n_gates += len(layer[1]) // 2

    return n_gates

def get_num_gates_qstring(q_string: str):
    '''
    Get the number of gates in a circuit represented by a string.
    input:
        :param q_string: string of quantum circuit with form e.g.
                         "H=0=nop=nop@XX=1=3=0.2"
    returns:
        the number of gates in the string
    '''
    return q_string.count("@") + 1

def break_qstr_to_gstrs(q_string:str):
    """
    This function breaks the string representation of a circuit into a list with
    string representations of the gates.

    Args:
        :q_string: A string representation of a tequila circuit object
    Returns:
        :list: A list of strings representing gates.
    Examples:
        >>> q_string = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        >>> g_list = break_qstr_to_gstrs(q_string)
        >>> print(g_list)
            ["H=0=nop=nop","X=1=nop=nop","RX=1=0=0.1","XX=0=3=0.2","XY=0=1=0.2"]
    """
    try:
        return list(q_string.split("@"))
    except:
        raise Exception("The string representation is incompatible")

def random_array(n_elem: int, distrib: str = "normal", rand_seed: int = None,
                 l_bound: float = 0, r_bound: float = 1,
                 mean: float = numpy.pi/2, scale: float = numpy.pi/4):
    '''
    Randomly generate a 1D array of ``n_elem`` elements.

    Args:
        :n_elem: The number of elements in the array.
    Kwargs:
        :distrib: Uniform distribution ("uniform") or normal distribution ("normal").
        :rand_seed: Random generator seed.
        :l_bound: Left bound for uniform distribution.
        :r_bound: Right bound for uniform distribution.
        :mean: Mean of the normal distribution.
        :scale: Standard deviation of the normal distribution.
    Returns:
        :numpy array: A randomly generated 1D array.
    '''
    assert distrib in ["normal", "uniform"], "Only 'normal' or 'uniform' distributions are supported!"
    numpy.random.seed(rand_seed)
    if distrib == "uniform":
        return numpy.random.uniform(l_bound, r_bound, n_elem)
    else:
        return numpy.random.normal(mean, scale, n_elem)

def random_chars(len_str: int = 4):
    '''
    Generate a random hashable string.
    Args:
        :len_str: the length of the string.
    Returns:
        :str: A randomly generated string with length len_str.
    '''
    str = random.choice(string.ascii_uppercase) # the first char is always a letter
    char_pool = string.ascii_uppercase + string.ascii_lowercase + string.digits
    for i in range(len_str-1):
        str += random.choice(char_pool)

    return str


def get_paulis_2q():
    '''
    Generate a list of all 2-qubit Pauli strings.
    '''
    paulis = ["X", "Y", "Z"]
    pauli_2q = []
    for p1 in paulis:
        for p2 in paulis:
            pauli_2q.append(p1 + p2)
    return pauli_2q

def cast_gate_2q_to_1q(gate_2q: str):
    '''
    Down cast a 2-qubit gate to a 1-qubit gate.

    Args:
        :gate_2q: Name of the 2-qubit gate.
    Returns:
        :str: Name of the 1-qubit gate.
    '''
    dict_2q_to_1q = __config__._cast_2q_to_1q
    paulis = ["X", "Y", "Z"]

    if gate_2q in dict_2q_to_1q:
        gate_1q = dict_2q_to_1q[gate_2q]
    elif len(gate_2q) == 2 and (gate_2q[0] in paulis) and (gate_2q[1] in paulis):
        gate_1q = "R" + gate_2q[1]
    else:
        raise ValueError("Cannot cast the 2-qubit gate to 1-qubit gate!")
    return gate_1q

def cast_gate_1q_to_2q(gate_1q: str):
    '''
    Change the single qubit gate to the corresponding two qubit gate.

    Args:
        :gate_1q: Name of the single-qubit gate.
    Returns:
        :str: Name of the 2-qubit gate (control gate).
    '''
    dict_1q_to_2q = __config__._cast_1q_to_2q
    paulis = ["X", "Y", "Z"]

    if gate_1q in dict_1q_to_2q:
        gate_2q = dict_1q_to_2q[gate_1q]

    else:
        raise ValueError("Cannot cast the 2-qubit gate to 1-qubit gate!")
    return gate_2q

def reverse_dict(dict_in: dict):
    '''
    Reverse the keys and values in a dictrionary.
    '''
    dict_out = {v: k for k, v in dict_in.items()}
    return dict_out

def count_qubit_symb_dict(symb_dict: dict):
    targ_qs = list(symb_dict[1])
    ctrl_qs = list(symb_dict[2])
    try:
        ctrl_qs.remove("nop")
    except:
        warnings.warn("Control qubit dictrionary has no nop!")
    targ_qs = numpy.asarray(targ_qs)
    ctrl_qs = numpy.asarray(ctrl_qs)
    n_qubit = max(max(targ_qs.astype(int)), max(ctrl_qs.astype(int)))
    return n_qubit + 1 # count starts at 0
