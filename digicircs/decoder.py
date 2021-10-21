'''
Converting a string into a Tequila quantum circuit.
Authors:
    Abhinav Anand <abhinav.anand@mail.utoronto.ca >
    Chong Sun <sunchong137@gmail.com>
    Jakob Kottmann <jakob.kottmann@utoronto.ca>
'''

import tequila as tq
import tequila.circuit.gates as tq_g
import warnings
from digicircs import __config__
from digicircs.utils import misc
import numpy
import numpy as np

# Dictionary of string to tequila gates
dict_string_to_tq = {'X': tq_g.X, 'Y': tq_g.Y, 'Z': tq_g.Z,
                    'H': tq_g.H, 'RX': tq_g.Rx, 'RY': tq_g.Ry,
                    'RZ': tq_g.Rz, 'ExpPauli':tq_g.ExpPauli,
                    'CRX': tq_g.CRx, 'CRY': tq_g.CRy, 'CRZ': tq_g.CRz,
                    'CNOT': tq_g.CNOT, 'CX': tq_g.CX, 'CY': tq_g.CY,
                    'CZ': tq_g.CZ, 'S': tq_g.S, 'T': tq_g.T}

# Default gates (Static and Parameterized)
DEFAULT_GATES = __config__._default_gates
SGATES_1Q = DEFAULT_GATES["_static_gates_for_1qubit"]
SGATES_2Q = DEFAULT_GATES["_static_gates_for_2qubits"]
PGATES_1Q = DEFAULT_GATES["_parameterized_gates_for_1qubit"]
PGATES_2Q = DEFAULT_GATES["_parameterized_gates_for_2qubit"]
GATES_1Q = SGATES_1Q + PGATES_1Q
GATES_2Q = PGATES_2Q + SGATES_2Q

def decoder(q_string:str, fix_params: bool=True, rm_ctrl: bool=True):
    """
    This function converts a string representation into its corresponding
    circuit

    Args:
        :q_string: a string representation of a tequila circuit object.
    Kwargs:
        :fix_param: If True - fix the parameters; if False - parameters taken as
                    variables.
        :rm_ctrl: If true, the one qubit gates cannot have control qubits.
    Returns:
        :tq.QCircuit: a Tequila circuit object converted from the string.
    Examples:
        >>> q_string = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        >>> tq_circ = decoder(q_string)
        >>> print(tq_circ)
            circuit:
            H(target=(0,))
            X(target=(1,))
            Rx(target=(1,), control=(0,), angle="whatever")
            Exp-Pauli(target=(0, 3), control=(), parameter=0.2, paulistring=X(0)X(3))
            Exp-Pauli(target=(0, 1), control=(), parameter=0.2, paulistring=X(0)Y(1))
    Notes:
        The case-sensitivity of gate name strings is removed: 'cnot' -> 'CNOT'.
        A warning will occur when the gate names are not included in the dict.
    """
    strings = misc.break_qstr_to_gstrs(q_string)

    q_circuit = tq.QCircuit()
    for string in strings:
        string = gate_preprocess(string, rm_ctrl=rm_ctrl)
        q_circuit += convert_string_to_gates(string, fix_params=fix_params)
    return q_circuit

def gate_preprocess(q_string: str, fix_params: bool=True, rm_ctrl:bool=True):
    """
    To make sure the one-to-one correspondance from string to gates, we consider
    the following exceptions:

    ============================================ ========================================
      Case                                       Interpretation
    ============================================ ========================================
      ``G`` is not provided.                     Invalid.
      ``T`` is not provided.                     Invalid.
     Single-qubit gate but ``C`` is defined.     Ignore ``C``.
     Two-qubit gate but ``C = T``.               Change to single-qubit gate.
     Two-qubit gate but ``C`` is not specified.  Change to single qubit gate.
     Static gates with parameters.               Ignore parameters.
     Parameterized gate but ``P`` not given.     Assign a random value or set as variable.
    ============================================ =========================================

    Args:
        :q_string: A string encoding the gate information.
    Kwargs:
        :fix_params: If true, the parameter must be a number, else can be a string.
        :rm_ctrl: If true, the one qubit gates cannot have control qubits.
    Returns:
        :str: The editted string following rules above.
    """
    # check if the strings are in the default gates.
    all_gates_list = GATES_1Q + GATES_2Q

    q_string = list(q_string.split("="))
    # get gate name, target qubit, control qubit, parameters.
    name    = q_string[0]
    target  = q_string[1]
    control = q_string[2]
    param   = q_string[3]

    if name not in all_gates_list + ["nop"]:
        raise Exception("Unknown gate name {} in q_string={}".format(name, q_string))

    if name in SGATES_1Q:
        if rm_ctrl:
            control = "nop"
        param = "nop"
    if name in PGATES_1Q:
        if rm_ctrl:
            control = "nop"
        if param == "nop": # TODO: check if we should turn PGATES into SGATES or assign a random one
            if fix_params:
                param = misc.random_array(1, distrib = "normal")[0]
                param = "{}".format(param)
            else:
                param = misc.random_chars(4)

    if name in SGATES_2Q:
        if control == "nop" or control == target:
            name = misc.cast_gate_2q_to_1q(name)
            control = "nop"
        param = "nop"
    if name in PGATES_2Q:
        if control == "nop" or control ==  target:
            name = misc.cast_gate_2q_to_1q(name)
            control = "nop"
        if param  == "nop":
            if fix_params:
                param = misc.random_array(1, distrib = "normal")[0]
                param = "{}".format(param)
            else:
                param = misc.random_chars(4)

    g_str = name + "=" + target + "=" + control + "=" + param
    return g_str

def qstring_preprocess(q_string: str, fix_params: bool = True):
    '''
    Preprocess a quantum string based on ``gate_preprocess``.
    Args:
        :q_string: A string encoding the quantum circuit.
    Kwargs:
        :fix_params: If true, the parameter must be a number, else can be a string.
    Returns:
        :str: The editted string following rules above.
    '''
    strings = misc.break_qstr_to_gstrs(q_string)
    q_string_edit = ""
    for g_str in strings:
        g_str_n = gate_preprocess(g_str, fix_params)
        #print(g_str_n)
        if g_str_n[:3] == "nop":
            g_str_n = ""
        q_string_edit += g_str_n + "@"
    return q_string_edit[:-1]


def convert_string_to_gates(q_string: str, fix_params: bool = True):
    """
    This function converts the string representation of a gate into a tequila
    gate object. The string representation of a gate has the syntax:

    ``"G=T=C=P"``

    where ``"G"`` is the gate name, ``"T"`` is the target qubit, ``"C"`` is the
    control qubit, and ``"P"`` is the parameter for parameterized gate.


    Args:
        :q_string:  A string representation of a tequila gate object.
    Kwargs:
        :fix_params: If True - fix the parameters; if False - parameters taken as
                    variables.
    Returns:
        :tq.QCircuit: A tequila object encoding the corresponding gate.
    Examples:
        >>> q_str = "H=0=nop=nop"
        >>> tq_gate = convert_string_to_gates(q_str)
        >>> print(tq_gate)
            H(target=(0,))
        >>> q_str = "RX=1=nop=0.1"
        >>> tq_gate = convert_string_to_gates(q_str)
        >>> print(tq_gate)
            Rx(target=(1,), parameter=0.1)
        >>> q_str = "CRX=0=1=0.2"
        >>> tq_gate = convert_string_to_gates(q_str)
        >>> print(tq_gate)
            Rx(target=(1,), control=(0,), parameter=0.2)
        >>> q_str = "CRX=0=1=a"
        >>> tq_gate = convert_string_to_gates(q_str, fix_params=False)
        >>> print(tq_gate)
            Rx(target=(1,), control=(0,), angle="a")
    """

    q_string = list(q_string.split("="))
    circuit = tq.QCircuit()


    # get gate name, target qubit, control qubit, parameters.
    name    = q_string[0]
    target  = q_string[1]
    control = q_string[2]
    param   = q_string[3]

    if name == "nop":
        warnings.warn("nop is provided for gate names.")
        return tq.QCircuit()
    else:
        name = name.upper()

    if target == "nop":
        raise Exception("No target given in q_string={}".format(q_string))
    else:
        target = int(target)

    if control == "nop":
        control = None
    else:
        control = int(control)

    if param == "nop":
        param = None
    else:
        try:
            param = float(param)
        except:
            param = str(param)

    try:
        if param is None:
            circuit += dict_string_to_tq[name](target = target,  control = control)
        else:
            try:
                circuit += dict_string_to_tq[name](target = target,  control = control, angle=param)
            except:
                circuit += dict_string_to_tq[name](target = target,  control = control, power=param)

    except:
        try:
            # ExpPauli
            paulistring = "{0}({1}){2}({3})".format(name[0], target, name[1], control)
            circuit += dict_string_to_tq["ExpPauli"](paulistring = paulistring, angle = param)

        except Exception as error:
            raise Exception("Error in q_string={}\n{}".format(name, error))

    return circuit
