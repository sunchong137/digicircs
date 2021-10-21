
"""Utility functions for circuit analysis"""

import cirq
import tequila as tq
from tequila.circuit.compiler import Compiler

# Default gates (Static and Parameterized)
SGATES_1Q = ["X", "Y", "Z", "H"]
SGATES_2Q = ["CNOT"]
PGATES_1Q = ["RX", "RY", "RZ"]
PGATES_2Q = ["CRX", "CRY", "CRZ"]
GATES_1Q = SGATES_1Q + PGATES_1Q
GATES_2Q = PGATES_2Q + SGATES_2Q

def count_cnots(circuit: tq.QCircuit):
    '''
    Counts number of CNOTs in given circuit.

    input:
        :param circuit: Circuit to analyze
    returns:
        :int: Number of CNOTs in circuit
    '''
    compiler = Compiler(trotterized=True, exponential_pauli=True, controlled_rotation=True)
    compiled = compiler(circuit)
    return sum([1 for g in compiled.gates if g.is_controlled() and g.name.lower()=="x"])

def compute_depth(circuit: tq.QCircuit):
    '''
    Compute depth of a given circuit.

    input:
        :param circuit: Circuit to analyze
    returns:
        :int: Circuit depth
    '''
    compiler = Compiler(trotterized=True, exponential_pauli=True, controlled_rotation=True)
    compiled = compiler(circuit)
    my_circuit = tq.compile(compiled, backend="cirq").circuit
    depth = len(cirq.Circuit(my_circuit.all_operations()))
    return depth

def compute_nparams(circuit: tq.QCircuit):
    '''
    Counts number of parameters in given circuit.

    input:
        :param circuit: Circuit to analyze
    returns:
        :int: Number of parameters
    '''
    nparams = 0
    for gate in circuit.gates:
        if gate.is_parametrized():
            nparams += 1
    return nparams

def compute_nmoments_from_qstr(q_str: str):
    '''
    Get the number of moments in a circuit represented by a string.

    Args:
        :q_str: A string encoding a quantum circuit
    Returns:
        :int: Number of moments in the quantum circuit.
    Example:
        >>> q_str = "Rz=0=nop=nop@Rz=4=nop=nop@CNOT=2=3=nop@Rz=1=nop=nop@Rz=5=nop=nop@CNOT=0=2=nop@CNOT=3=4=nop"
        >>> n_moments = compute_nmoments_from_qstr(q_str)
        >>> print(n_moments)
            2
    '''
    q_list = q_str.split("@")
    qubit_count = []
    n_moments = 1
    for gate in q_list:
        g_elems = gate.split("=")
        try:
            _gname = g_elems[0]
            _targ = int(g_elems[1])
            _ctrl = g_elems[2]
            if _ctrl != "nop":
                _ctrl = int(_ctrl)
            else:
                _ctrl = None
            if _gname in GATES_1Q:
                _ctrl = None #ignore control qubit for 1-qubit gates
        except:
            raise ValueError("The string given is in valid!")

        if _ctrl is None:
            if _targ in qubit_count:
                n_moments += 1
                qubit_count = []
            qubit_count += [_targ]
        else:
            if _targ in qubit_count:
                n_moments += 1
                qubit_count = []
            elif _ctrl in qubit_count:
                n_moments += 1
                qubit_count = []
            qubit_count += [_targ, _ctrl]

    return n_moments

def edit_qpic_file(file_to_modify, tq_circuit, file_to_save='temp.qpic'):
    '''
    Edits circuit diagram so that parameterized gates are
    pink (vs. blue).

    input:
        :file_to_modify: qpic file to modify
        :tq_circuit: tequila circuit to draw
        :file_to_save: qpic file to save. Writes to temp.qpic if undefined.
    '''
    # Obtain indices of parameterized gates from tequila circuit
    p_ind = []
    for i, g in enumerate(tq_circuit.gates):
        try:
            p = g.parameter()
            p_ind.append(i)
        except:
            pass

        try:
            p = g.is_parameterized()
            p_ind.append(i)
        except:
            pass

    with open(file_to_modify) as reader:
        data = reader.readlines()

    # Lines + indices for gates from qpic file
    gate_ind = [i for i, d in enumerate(data) if "P:fill=tq" in d]
    gate_ops = [d for d in data if "P:fill=tq" in d]

    for i in range(len(gate_ops)):
        # Change color
        if i in p_ind:
            gate_ops[i] = gate_ops[i].replace('tq', 'guo')

        # Change font
        t = gate_ops[i].replace('\\textcolor{white}{', '\\textcolor{white}{\\textsf{')
        t = t.replace('} ', '}} ')
        gate_ops[i] = t

    ii = 0
    for i in range(len(data)):
        if i in gate_ind:
            data[i] = gate_ops[ii]
            ii+= 1

    if file_to_save is None:
        file_to_save = file_to_modify

    with open(file_to_save, 'w') as writer:
        writer.writelines(data)


def simplify_qstring(qstr: str):
    '''
    Simplify the circuit string based on Tequila syntax.
        CNOT           -> X with control qubit.
        CRX, CRY, CRZ  -> RX, RY, RZ with control qubit.
    '''
    qstr = qstr.replace("CNOT", "X")
    qstr = qstr.replace("CRX", "RX")
    qstr = qstr.replace("CRY", "RY")
    qstr = qstr.replace("CRZ", "RZ")

    return qstr
