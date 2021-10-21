'''
Convert a Tequila quantum circuit into a string.

Authors:
    Abhinav Anand <abhinav.anand@mail.utoronto.ca>
    Chong Sun <sunchong137@gmail.com>
    Jakob Kottmann <jakob.kottmann@utoronto.ca>
'''
import tequila as tq
import numpy as np
from digicircs import __config__
from digicircs.utils import misc
from tequila.circuit.compiler import Compiler

# Default gates (Static and Parameterized)
DEFAULT_GATES = __config__._default_gates
SGATES_1Q = DEFAULT_GATES["_static_gates_for_1qubit"]
SGATES_2Q = DEFAULT_GATES["_static_gates_for_2qubits"]
PGATES_1Q = DEFAULT_GATES["_parameterized_gates_for_1qubit"]
PGATES_2Q = DEFAULT_GATES["_parameterized_gates_for_2qubit"]
GATES_1Q = SGATES_1Q + PGATES_1Q
GATES_2Q = PGATES_2Q + SGATES_2Q

def encoder(circuit :tq.QCircuit):
    """
    This function converts a circuit into a string representation.
    Tequila gate list: <https://aspuru-guzik-group.github.io/tequila/package/circuit/tequila.circuit.gates.html#module-tequila.circuit.gates>
    Args:
        :circuit: a Tequila circuit object.
    Returns:
        :str: a string representing the circuit.
    Examples:
        >>> circuit = tq.gates.H(target=0) + tq.gates.CRX(target=1,control=0,angle=0.1) + tq.gates.ExpPauli("X(0)X(3)",angle=0.2)
        >>> q_str = encoder(circuit)
        >>> print(q_str)
        H=0=nop=nop@RX=1=0=0.1@XX=0=3=0.2
    """
    # digicircs encoder does not support all tequila gate types
    # multicontrol works on latest branch (not on pypi version) ... don't use it for now
    try:
        gates = _break_circuit(circuit)
        q_string = ""
        for gate in gates:
            q_string += _convert_gates_to_string(gate)
            q_string += "@"
    except:
        compiler=Compiler(exponential_pauli=True, multicontrol=False,
                          trotterized=True, generalized_rotation=True,
                          controlled_exponential_pauli=True, power=True,
                          controlled_power=True, hadamard_power=True,
                          toffoli=True, controlled_phase=True, phase=True,
                          phase_to_z=True)

        circuit = compiler(circuit)
        gates = _break_circuit(circuit)
        q_string = ""
        for gate in gates:
            q_string += _convert_gates_to_string(gate)
            q_string += "@"


    return q_string[:-1]

def _break_circuit(circuit :tq.QCircuit):
    """
    This function returns a list with all the gates in the circuit

    Args:
        :circuit: A tequila circuit object
    Returns:
        :list: a list of tequila gate objects.
    Examples:
        >>> circuit = tq.gates.H(target=0) + tq.gates.X(target=1)
                    + tq.gates.CRx(target=1,control=0,angle=0.1)
                    + tq.gates.ExpPauli("X(0)X(3)",0.2)
                    + tq.gates.ExpPauli("X(0)Y(1)",0.2)
        >>> gates = _break_circuit(circuit)
        >>> print(gates)
            [tq.gates.H(target=0), tq.gates.X(target=1), tq.gates.CRx(target=1,control=0,angle=0.1),
             tq.gates.ExpPauli("X(0)X(3)",0.2), tq.gates.ExpPauli("X(0)Y(1)",0.2)]]
    """
    try:
        return circuit.gates
    except:
        raise Exception("The circuit structure is incompatible")

def _convert_gates_to_string(gate: tq.gates):
    """
    This function returns the string corresponding to a gate in the form
    ``"<name>=<target>=<control>=<parameter>"``, and uses "nop" to denote the
    absence of a functionality.

    current list of gates supported: "X", "Y", "Z", "H", "S", "T",
                                     "Rx", "Ry", "Rz",
                                     "CRx", "CRy", "CRz", "CNOT",
                                     "CX", "CY", "CZ",
                                     ExpPauli gates (XX, YY, ZZ, and other variants).

    Note: not all tequila gates are supported.

    Args:
        :gate: A tequila gate object.
    Returns:
        :str: A string encoding the given gate.
    Examples:
        >>> gate1 = tq.gates.H(target=0).gates[0]
        >>> str2 = _convert_gates_to_string(gate1)
        >>> print(str2)
            "H=0=nop=nop"
        >>> gate2 = tq.gates.Rx(target=1,angle=0.1).gates[0]
        >>> str2 = _convert_gates_to_string(gate2)
        >>> print(str2)
            "RX=1=nop=0.1"
        >>> gate3 = tq.gates.CRx(target=1,control=0,angle=0.1).gates[0]
        >>> str3 = _convert_gates_to_string(gate3)
        >>> print(str3)
            "CRX=1=0=0.1"
        >>> gate4 = tq.gates.ExpPauli("X(0)X(3)",0.2).gates[0]
        >>> str4 = _convert_gates_to_string(gate4)
        >>> print(str4)
            "XX=0=3=0.2"
    """
    name = "nop"
    target = "nop"
    control = "nop"
    param = "nop"

    try:
        #constructing the string for Exponential Pauli Gate
        # --> circumvented by compiler above, does not have a working decoder
        pauli = []
        qubits = []
        for k,v in gate.paulistring._data.items():
            pauli.append(v)
            qubits.append(k)
        name = "".join(pauli)
        return name+"="+str(qubits[0])+"="+str(qubits[1])+"="+str(gate.parameter)
    except:
        #for every other gate
        name = str(gate.name).upper()
        target = list(gate.target)[0]

    try:
        #for gate with control
        control = list(gate.control)[0]
    except:
        pass
    if name in GATES_1Q and control != "nop":
        name = misc.cast_gate_1q_to_2q(name)
    try:
        #for parameterized gate
        param = gate.parameter
    except:
        pass

    # avoid naming parameters as objectives
    if hasattr(param, "extract_variables"):
        param = param.extract_variables()
        if len(param) == 1:
            param = param[0]

    return name+"="+str(target)+"="+str(control)+"="+str(param)


if __name__ == "__main__":
    from one_hot import *
    from decoder import *
    # Create a circuit in tequila
    circuit = tq.gates.H(target=0) + tq.gates.X(target=1)+ tq.gates.CRx(target=1,control=0,angle=0.1) + tq.gates.ExpPauli("X(0)X(3)",0.2) +tq.gates.ExpPauli("X(0)Y(1)",0.2)
    print("circuit to be encoded")
    tq.draw(circuit)
    # Create the string representation of the circuit
    circuit_string = encoder(circuit)
    print("string representation of the circuit")
    print(circuit_string)
    # create the dataet of circuits
    dataset = [circuit_string]
    print("dataset of the circuit strings")
    print(dataset)
    # get the uniques gates, control and target qubits from the dataset and the maximum number of gates
    # in a circuit in the dataset
    gates,target,control,max_len = get_symbols_from_qstring_list(dataset)
    print("unique gates, unique targets, unique controls, maximum gates number")
    print(gates,target,control,max_len)
    # create the dictionaries for the one hot encoding and for the reverse decoding
    symbol_dictionary_list, reverse_e_dictionary_list = (create_symbol_dictionary([gates,target,control]))
    print("Encoding dictionary")
    print(symbol_dictionary_list)
    print("Decoding dictionary")
    print(reverse_e_dictionary_list)
    # create the list of default unary string
    unary_string_list = get_unary_string(symbol_dictionary_list)
    print("default unary string")
    print(unary_string_list)
    # get the decimal_encoding and one_hot_encoding of the string representation pf a circuit
    decimal_encoding, one_hot_encoding = to_one_hot(circuit_string,max_len,symbol_dictionary_list,unary_string_list)
    print("Decimal encoding")
    print(decimal_encoding)
    print("One hot encoding")
    print(one_hot_encoding)
    # get the circuit string from the one hot encoding
    circuit_string = from_one_hot(one_hot_encoding, reverse_e_dictionary_list)
    print("the decoded circuit string")
    print(circuit_string)
    # convert the string representation to the circuit object
    circuit = decoder(circuit_string)
    print("The decoded circuit")
    tq.draw(circuit)
