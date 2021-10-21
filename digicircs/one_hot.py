#import tequila as tq
#import numpy as np
import copy

def _break_strings(q_string: str):
    """
    This function finds the unique gate symbols in the string representation
    of a quantum circuit.

    Args:
        :q_string: A string representation of a tequila circuit object
    Returns:
        :set: A set of symbols representing the unique gate names in q_string.
        :set: A set of symbols representing the unique target qubits.
        :set: A set of symbols representing the unique target qubits.
    Examples:
        >>> q_string = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        >>> name, target, control = _break_strings(q_string)
        >>> print(name, target, control)
            {'X', 'RX', 'H', 'XY', 'XX'}, {'0', '1'}, {'3', '0', 'nop', '1'}
    """
    name = set()
    target = set()
    control = set()
    for element in list(q_string.split("@")):
        g_string = list(element.split("="))
        name.add(g_string[0])
        target.add(g_string[1])
        control.add(g_string[2])
    return name, target, control

def get_symbols_from_qstring_list(q_strings: list):
    """
    This function finds the unique symbols in a list of string
    representations of different tequila circuit objects, and
    also the maximum number of gates in a circuit object

    Args:
        :q_strings:  A list of string representations of different tequila circuit objects.
    Returns:
        :set: A set of gate names.
        :set: A set of target qubits (in strings).
        :set: A set of control qubits (in strings).
        :int: the maximum number of gates.
    Examples:
        >>> q_string = ["H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2",
                        "Y=0=nop=nop@X=1=nop=nop@RY=1=0=0.1@ZZ=0=3=0.2@XY=0=1=0.2"]
        >>> names, targets, controls, max_len = get_symbols_from_qstring_list(q_string)
        >>> print(names, targets, controls, max_len)
            {'X', 'RX', 'H', 'XY', 'XX', 'Y', 'ZZ'}, {'0', '1'}, {'3', '0', 'nop', '1'}, 5
    """
    # TODO check the returns in the example
    names = set()
    targets = set()
    controls = set()
    max_len = 0
    for q_string in q_strings:
        name, target, control = _break_strings(q_string)
        n_gate = len(list(q_string.split("@")))
        if n_gate > max_len:
            max_len = n_gate
        names = names | name
        targets = targets | target
        controls = controls | control
    return names, targets, controls, max_len

def create_symbol_dictionary(elements: list):
    """
    This function converts sets of different strings into a dictionary by giving
    every string a unique integer value, and adds an element "nop" to every
    dictionary if already not present, and also creates a reverse dictionary for
    getting the strings too.

    We add "nop" to control qubit set.

    Args:
        :elements: a list of set of unique string values
    Returns:
        :dict: the dictionary of symbols to numbers.
        :dict: the reverse dictionary of numbers to symbols.
    Examples:
        >>> elements = [{'X', 'RX', 'H', 'XY', 'XX', 'Y', 'ZZ'}, {'0', '1'}, {'3', '0', 'nop', '1'}]
        >>> symbol_dict, rev_symbol_dict = create_symbol_dictionary(elements)
        >>> print(symbol_dict)
            [{'X': 0, 'XY': 1, 'H': 2, 'RX': 3, 'RY': 4, 'Y': 5, 'XX': 6, 'ZZ': 7},
             {'0': 0, '1': 1}, {'3': 0, '0': 1, 'nop': 2, '1': 3}]
        >>> print(rev_symbol_dict)
            [{0: 'X', 1: 'XY', 2: 'H', 3: 'RX', 4: 'RY', 5: 'Y', 6: 'XX', 7: 'ZZ'},
             {0: '0', 1: '1'}, {0: '3', 1: '0', 2: 'nop', 3: '1'}]
    """
    #TODO check symbol and reverse dictionaries in the example
    symbol_dictionary_list = []
    reverse_dictionary_list = []
    elements[0].add("nop")
    elements[-1].add("nop")
    for element in elements:
        element_dictionary = {key:i for i,key in enumerate(element)}
        reverse_e_dictionary = {i:key for i,key in enumerate(element)}
        symbol_dictionary_list.append(element_dictionary)
        reverse_dictionary_list.append(reverse_e_dictionary)
    return symbol_dictionary_list,reverse_dictionary_list

def get_unary_string(symbol_dictionary_list: list):
    """
    This function creates a list of all 0s of length equal to the length
    of unique keys in the dictionary

    Args:
        :symbol_dictionary_list: a list of dictionaries.
    Returns:
        :list: a list of 0s with the same shape of one-hot encoding strings.
    Examples:
        >>> symbol_dict_list = [{'X': 0, 'XY': 1, 'H': 2, 'RX': 3, 'RY': 4, 'Y': 5}, {'0': 0, '1': 1}, {'3': 0, '0': 1, 'nop': 2, '1': 3}]
        >>> unary_lst = get_unary_string(symbol_dict_list)
        >>> print(unary_lst)
            [[0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0]]
    """
    unary_string_list = []
    for element in symbol_dictionary_list:
        unary_string = []
        for _ in range(len(list(element.keys()))):
            unary_string.append(0)
        unary_string_list.append(unary_string)
    return unary_string_list


def to_one_hot(q_string:str, max_len:int, symbol_dictionary:list,
               zero_unary_strings:list, encode_params: bool = True):
    """
    This function converts the string representation of a circuit to a
    one hot encoding and a decimal_encoding using the dictionary "symbol_dictionary",
    with the length of the list equal to "max_len"

    Args:
        :q_string: A string representation of a tequila circuit object
        :max_len: The maximum number of gates in circuit in the dataset
        :symbol_dictionary: A list of dictionaries with the keys as (gate symbols,
                              control and target values) and values as unique integer values
        :zero_unary_strings: A list of list of default unary string for ciruits
    Kwargs:
        :encode_params: if True, the parameters are included.
    Returns:
        :list: decimal encoding
        :list: one-hot encoding
    Examples:
        >>> q_string = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XY=0=3=0.2@RY=0=1=0.2"
        >>> max_len = 5
        >>> symbol_dictionary = [{'RY': 0, 'XY': 1, 'X': 2, 'Y': 3, 'RX': 4, 'H': 5, 'nop':6},
                         {'0': 0, '1': 1}, {'nop': 0, '1': 1, '0': 2, '3': 3}]
        >>> zero_unary_strings = [[0, 0, 0, 0, 0, 0], [0, 0 ], [0, 0, 0, 0]]
        >>> decimal_encoding, one_hot_encoding = to_one_hot(q_string, max_len, symbol_dictionary, zero_unary_strings)
        >>> print(decimal_encoding)
            [[5, 0, 0, 0.3], [2, 1, 0, 0.4], [4, 1, 2, 0.1], [1, 0, 3, 0.2], [0, 0, 1, 0.2]]
        >>> print(one_hot_encoding)
            [[[0, 0, 0, 0, 0, 1, 0], [1, 0], [1, 0, 0, 0], 0.3],
             [[0, 0, 1, 0, 0, 0, 0], [0, 1], [1, 0, 0, 0], 0.4],
             [[0, 0, 0, 0, 1, 0, 0], [0, 1], [0, 0, 1, 0], 0.1],
             [[0, 1, 0, 0, 0, 0, 0], [1, 0], [0, 0, 0, 1], 0.2],
             [[1, 0, 0, 0, 0, 0, 0], [1, 0], [0, 1, 0, 0], 0.2]]
    """
    ohe = []
    decimal_encoding = []
    for elements in list(q_string.split("@")):
        temp_list = []
        d_temp_list = []
        element_list = list(elements.split("="))
        for ind, element in enumerate(element_list[:-1]):
            value = symbol_dictionary[ind][element]
            d_temp_list.append(value)
            temp_u = copy.deepcopy(zero_unary_strings[ind])
            temp_u[value]=1
            temp_list.append(temp_u)

        if encode_params:
            try:
                #print((list(elements.split("="))[-1]))
                temp_list.append(float(element_list[-1]))
                d_temp_list.append(float(element_list[-1]))
            except:
                temp_list.append(0.2)
                d_temp_list.append(0.2)

        ohe.append(temp_list)
        decimal_encoding.append(d_temp_list)

    pad_len = max_len - len(ohe)

    for _ in range(pad_len):
        temp_list = []
        d_temp_list = []
        for ind in range(3):
            try:
                value = symbol_dictionary[ind]["nop"]
            except:
                value = 0 # target has no "nop"
            d_temp_list.append(value)
            temp_u = copy.deepcopy(zero_unary_strings[ind])
            temp_u[value]=1
            temp_list.append(temp_u)
        if encode_params:
            temp_list.append(0.2)
            d_temp_list.append(0.2) # change it later
            
        ohe.append(temp_list)
        decimal_encoding.append(d_temp_list)
    return decimal_encoding, ohe

def from_one_hot(ohe_string, reverse_e_dictionary_list):
    """
    This function converts the one hot encoding of a circuit string into
    a string representation of the circuit

    Args:
        :ohe_string: one hot encoding of the full circuit
        :reverse_e_dictionary_list: a dictionary with the key as the unique integer
                                      value for the gate and the value as the gate identifier
                                      usually the reverse dictionary used for encoding
    Returns:
        :str: a string representing the circuit.
    Examples:
        >>> ohe_string=[[[0, 0, 0, 0, 0, 1], [1, 0], [1, 0, 0, 0], 0.3],
                        [[0, 0, 1, 0, 0, 0], [0, 1], [1, 0, 0, 0], 0.4],
                        [[0, 0, 0, 0, 1, 0], [0, 1], [0, 0, 1, 0], 0.1],
                        [[0, 1, 0, 0, 0, 0], [1, 0], [0, 0, 0, 1], 0.2],
                        [[1, 0, 0, 0, 0, 0], [1, 0], [0, 1, 0, 0], 0.2]]
        >>> reverse_e_dictionary_list=[{0:'RY', 1:'XY', 2:'X', 3:'Y', 4:'RX', 5:'H'},
                         { 0:'0', 1:'1'}, {0:'nop', 1:'1', 2:'0', 3:'3'}]
        >>> q_string = from_one_hot(ohe_string, reverse_e_dictionary_list)
        >>> print(q_string)
            H=0=nop=0.3@X=1=nop=0.4@RX=1=0=0.1@XY=0=3=0.2@RY=0=1=0.2
    """
    q_string = ""
    for gate_encoding in ohe_string:
        for ind,element in enumerate(gate_encoding[:-1]):
            position=element.index(1)
            q_string += (reverse_e_dictionary_list[ind][position])
            q_string += "="
        q_string = q_string[:-1] + "=" + str(gate_encoding[-1]) +"@"
    return q_string[:-1]



def _compare_one_hot(ohe1, ohe2):
    '''
    compare if two one hot encoding lists are equal
    input:
        :type ohe1: list of three lists and one float
        :type ohe2: list of three lists and one float
        :param ohe1: one hot encoding of a circuit
        :param ohe2: one hot encoding of a circuit
    returns:
        a boolean value
    '''

    ohe_len = len(ohe1)
    eq = 1
    for i in range(ohe_len):
        eq *= int(ohe1[i] == ohe2[i])
    return eq
