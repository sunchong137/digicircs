
# default gate names used in the module
_default_gates = {
    "_static_gates_for_1qubit" : ["X", "Y", "Z", "H"], \
    "_static_gates_for_2qubits": ["CNOT"], \
    "_parameterized_gates_for_1qubit": ["RX", "RY", "RZ"],\
    "_parameterized_gates_for_2qubit": ["CRX", "CRY", "CRZ", "XX", "XY", "XZ",  "YY", "YZ", "ZZ"]
    }

_cast_2q_to_1q = {"CNOT": "X", "CRX": "RX", "CRY": "RY", "CRZ": "RZ"}
_cast_1q_to_2q = {"X": "CNOT", "RX": "CRX", "RY": "CRY", "RZ": "CRZ"}

#PAULIS = ["X", "Y", "Z"]

# default gate names used in the module
#_default_gates = {
#    "_static_gates_for_1qubit" : [], \
#    "_static_gates_for_2qubits": [], \
#    "_parameterized_gates_for_1qubit": ["RX"],\
#    "_parameterized_gates_for_2qubit": ["XY"]
#    }
#
#_cast_2q_to_1q = {"XY": "RX"}
#_cast_1q_to_2q = {"RX": "XY"}
