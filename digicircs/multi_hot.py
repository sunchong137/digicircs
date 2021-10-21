'''
Functions to convert between quantum circuit strings and multi-hot encoding.
String encoding   : ``G1=T1=C1=P1@G2=T2=C2=P2`` for a two-gate circuit.
Multi-hot encoding: [[1, 0, 1, 0, 1, 0, 0, 0.2],
                     [0, 1, 0, 1, 0, 0, 1, 0.3]]
                    where the first two bits corresponds to the gate names,
                    the 3rd and 4th bits corresponds to the target qubits,
                    the 5th-7th bits corresponds to the control qubits,
                    and the last bit correspond to the parameter.
Author: Chong Sun <sunchong137@gmail.com>
'''
import copy
import numpy
import torch
from digicircs import one_hot

def _break_strings(q_string: str):
    return one_hot._break_strings(q_string)

def get_symbols_from_qstring_list(q_strings: list):
    return one_hot.get_symbols_from_qstring_list(q_strings)

def create_symbol_dictionary(elements: list):
    return one_hot.create_symbol_dictionary(elements)

def get_unary_string(symbol_dictionary_list: list):
    return one_hot.get_unary_string(symbol_dictionary_list)


def to_multi_hot(q_string:str, max_len:int, symbol_dictionary:list,
                 zero_unary_strings:list, encode_params: bool = True):
    '''
    Generate a multi-hot representation of the quantum circuit.

    Args:
        :q_string: A string representation of a tequila circuit object
        :max_len: The maximum number of gates in circuit in the dataset
        :symbol_dictionary: A list of dictionaries with the keys as (gate symbols,
                              control and target values) and values as unique integer values
        :zero_unary_strings: A list of list of default unary string for ciruits.
    Kwargs:
        :encode_params: if True, the parameters are also encoded.
    Returns:
        :list: decimal encoding
        :list: multi-hot encoding of the circuit string.
    '''
    decimal_encoding, ohe = one_hot.to_one_hot(q_string, max_len,
                                               symbol_dictionary,
                                               zero_unary_strings,
                                               encode_params=encode_params)
    mhe = []
    if encode_params:
        for _g in ohe:
            mhe.append(list(_g[0] + _g[1] + _g[2] + [_g[3]]))
    else:
        for _g in ohe:
            mhe.append(list(_g[0] + _g[1] + _g[2]))

    return decimal_encoding, mhe

def from_multi_hot(mhe_string: list, reverse_e_dictionary_list: dict,
                   encode_params: bool = True):
    """
    This function converts the multi-hot encoding (with or without moise)
       of a circuit string into a string representation of the circuit

    Args:
        :mhe_string: multi-hot encoding of the full circuit
        :reverse_e_dictionary_list: a dictionary with the key as the unique integer
                                      value for the gate and the value as the gate identifier
                                      usually the reverse dictionary used for encoding
    Kwargs:
        :encode_params: if True, the parameters are also included.
    Returns:
        :str: a string representing the circuit.
    Examples:
        >>> mhe_string=[[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0.3],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0.4],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0.1],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.2],
                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.2]]
        >>> reverse_e_dictionary_list=[{0:'RY', 1:'XY', 2:'X', 3:'Y', 4:'RX', 5:'H', 6: 'nop'},
                         { 0:'0', 1:'1'}, {0:'nop', 1:'1', 2:'0', 3:'3'}]
        >>> q_string = from_one_hot(mhe_string, reverse_e_dictionary_list)
        >>> print(q_string)
            H=0=nop=0.3@X=1=nop=0.4@RX=1=0=0.1@XY=0=3=0.2@RY=0=1=0.2

    """
    q_string = ""
    n_gates = len(reverse_e_dictionary_list[0])
    n_targets = len(reverse_e_dictionary_list[1])
    n_controls = len(reverse_e_dictionary_list[2])

    if type(mhe_string) is list:
        mhe_string = numpy.array(mhe_string)
    ct = 0
    for gate_encoding in mhe_string:
        ind_g = int(gate_encoding[:n_gates].argmax())
        ind_t = int(gate_encoding[n_gates: (n_gates+n_targets)].argmax())
        ind_c = int(gate_encoding[(n_gates+n_targets):(n_gates+n_targets+n_controls)].argmax())

        gate_name = reverse_e_dictionary_list[0][ind_g]
        targ_q = reverse_e_dictionary_list[1][ind_t]
        ctrl_q = reverse_e_dictionary_list[2][ind_c]
        g_string = gate_name + "=" + targ_q + "=" + ctrl_q + "="
        if encode_params:
            g_string += str(float(gate_encoding[-1]))
        else:
            g_string += "nop%d"%ct
            ct += 1
        q_string += g_string + "@"
    return q_string[:-1]

def add_noise_to_mhe(mhe: list, upper_bound: float, encode_params: bool = True,
                     rand_seed: int = None):
    """
    Replaces all zeroes with a random float in the range [0,upper_bound].

    Args:
        :mhe: the multi-hot list, numpy or torch.
        :upper_bound: upper bound of the random float range.
        :encode_param: if True, the parameters are not editted.
        :rand_seed: random seed, for test purpose only!
    Returns:
        :ndarray: the edited multi-hot encoding array.
    """
    array_type = type(mhe)
    if array_type is numpy.ndarray:
        numpy.random.seed(rand_seed)
        noise = upper_bound * numpy.random.rand(*mhe.shape)
    elif array_type is torch.Tensor:
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        noise = upper_bound * torch.rand(mhe.shape)

    if encode_params:
        noise[...,-1] = 0. # no noise to parameter
    new_mhe = mhe + noise
    if encode_params:
        new_mhe[..., :-1][new_mhe[..., :-1] > 1] = 1
    else:
        new_mhe[new_mhe > 1] = 1
    return new_mhe

def remove_noise_mhe(mhe:list, encode_param: bool = True):
    '''
    Turn a noisy multi-hot encoding into the standard multi-hot encoding.

    Args:
        :mhe: the noisy multi-hot list, numpy or torch.
        :encode_param: if True, the parameters are not editted.
    Returns:
        :ndarray: the standard multi-hot encoding array.
    '''
    params = copy.copy(mhe[...,-1]) # save the parameters
    array_type = type(mhe)
    if array_type is numpy.ndarray:
        dtype = mhe.dtype
        new_mhe = mhe.astype(int).astype(dtype)
    elif array_type is torch.Tensor:
        dtype = mhe.dtype
        new_mhe = mhe.type(torch.int)  .type(dtype)
    new_mhe[...,-1] = params

    return new_mhe
