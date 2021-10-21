#import pytest
import numpy as np
from digicircs.utils import misc, circ_utils, dd_utils

class TestMisc():
    '''
    Tests for functions in misc.py
    '''

    def test_ravel_list(self):
        lst = [[[0,1],[1,2,3], []], [[1,2,3,4]]]
        out_lst = misc.ravel_list(lst)
        ref_lst = [0,1,1,2,3,1,2,3,4]
        assert out_lst == ref_lst

    def test_get_num_gates_topo(self):
        circ_topo = [[[0], [1,2,3,5]], [[1,4], [2,5]]]
        out_len = misc.get_num_gates_topo(circ_topo)
        assert out_len == 6

    def test_get_num_gates_qstring(self):
        q_str = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@RY=0=1=0.2"
        n_gates = misc.get_num_gates_qstring(q_str)
        assert n_gates == 4

    def test_break_string(self):
        circuit_str = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        gate_strs = misc.break_qstr_to_gstrs(circuit_str)
        assert gate_strs[0] == "H=0=nop=nop"
        assert gate_strs[1] == "X=1=nop=nop"
        assert gate_strs[2] == "RX=1=0=0.1"
        assert gate_strs[3] == "XX=0=3=0.2"
        assert gate_strs[4] == "XY=0=1=0.2"

    def test_reverse_dict(self):
        dict = {"a":1, "b":2, "c":3}
        rev_dict = misc.reverse_dict(dict)
        rev_dict_ref = {1:"a", 2:"b", 3:"c"}
        assert rev_dict == rev_dict_ref

    def test_count_qubit_symb_dict(self):
        symb_dict = [{'X': 0, 'XY': 1, 'H': 2, 'RX': 3, 'Y': 4, 'XX': 5, 'ZZ': 6, 'nop': 7},
                    {'0': 0, '1': 1}, {'3': 0, '0': 1, 'nop': 2, '1': 3}]
        n_qubit = misc.count_qubit_symb_dict(symb_dict)
        assert n_qubit == 4

class TestCircUtils():
    '''
    Tests for functions in circ_utils.py
    '''
    def test_compute_nmoments_from_qstr(self):
        q_str1 = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@RY=0=1=0.2"
        out_1 = circ_utils.compute_nmoments_from_qstr(q_str1)
        assert out_1 == 2 # control qubits for 1-q gates are ignored
        q_str2 = "CRY=0=1=3.2401@Z=0=nop=0.7083@RX=1=nop=1.7771@CRX=2=0=0.2264\
                  @CRX=1=0=0.6219@RZ=1=nop=0.7584@CRZ=1=0=0.7685@CRZ=1=0=1.4159\
                  @CRX=1=0=1.3606@Z=1=nop=1.0368"
        out_2 = circ_utils.compute_nmoments_from_qstr(q_str2)
        assert out_2 == 9
