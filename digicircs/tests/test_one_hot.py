#import pytest
import numpy as np
from digicircs import one_hot

class TestOneHot():
    '''
    tests for functions in one_hot.py
    '''
    def test_break_strings(self):
        q_string = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        name, target, control = one_hot._break_strings(q_string)
        assert set(name) == set({'X', 'RX', 'H', 'XY', 'XX'})
        assert set(target) == set({'0', '1'})
        assert set(control) == set({'3', '0', 'nop', '1'})

    def test_get_symbols_from_qstring_list(self):
        q_string = ["H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2",
                    "Y=0=nop=nop@X=1=nop=nop@RY=1=0=0.1@ZZ=0=3=0.2@XY=0=1=0.2"]
        names, targets, controls, max_len = one_hot.get_symbols_from_qstring_list(q_string)
        assert set(names) == set({'X', 'RX', 'RY', 'H', 'XY', 'XX', 'Y', 'ZZ'})
        assert set(targets) == set({'0', '1'})
        assert set(controls) == set({'3', '0', 'nop', '1'})
        assert max_len == 5

    def test_create_symbol_dictionary(self):
        elements = [{'X', 'RX', 'H', 'XY', 'XX', 'Y', 'ZZ'}, {'0', '1'}, {'3', '0', 'nop', '1'}]
        sym_dicts, rev_dicts = one_hot.create_symbol_dictionary(elements)
        ref_sym_dicts = [{'X': 0, 'XY': 1, 'H': 2, 'RX': 3, 'Y': 4, 'XX': 5,
            'ZZ': 6, 'nop': 7},
                        {'0': 0, '1': 1}, {'3': 0, '0': 1, 'nop': 2, '1': 3}]
        ref_rev_dicts = [{0: 'X', 1: 'XY', 2: 'H', 3: 'RX', 4: 'Y', 5: 'XX', 6:
            'ZZ', 7: 'nop'}, \
                        {0: '0', 1: '1'}, {0: '3', 1: '0', 2: 'nop', 3: '1'}]

        # compare keys
        assert set(sym_dicts[0]) == set(ref_sym_dicts[0])
        assert set(sym_dicts[1]) == set(ref_sym_dicts[1])
        assert set(sym_dicts[2]) == set(ref_sym_dicts[2])
        assert set(rev_dicts[0]) == set(ref_rev_dicts[0])
        assert set(rev_dicts[1]) == set(ref_rev_dicts[1])
        assert set(rev_dicts[2]) == set(ref_rev_dicts[2])
        # compare values
        assert set(sym_dicts[0].values()) == set(ref_sym_dicts[0].values())
        assert set(sym_dicts[1].values()) == set(ref_sym_dicts[1].values())
        assert set(sym_dicts[2].values()) == set(ref_sym_dicts[2].values())
        assert set(rev_dicts[0].values()) == set(ref_rev_dicts[0].values())
        assert set(rev_dicts[1].values()) == set(ref_rev_dicts[1].values())
        assert set(rev_dicts[2].values()) == set(ref_rev_dicts[2].values())

    def test_get_unary_string(self):
        sym_dicts = [{'X': 0, 'XY': 1, 'H': 2, 'RX': 3, 'RY': 4, 'nop': 5, 'Y': 6, 'XX': 7, 'ZZ': 8}, \
                     {'0': 0, '1': 1, 'nop': 2}, {'3': 0, '0': 1, 'nop': 2, '1': 3}]

        unary_strs = one_hot.get_unary_string(sym_dicts)
        assert set(unary_strs[0]) == set([0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert set(unary_strs[1]) == set([0, 0, 0])
        assert set(unary_strs[2]) == set([0, 0, 0, 0])

    def test_to_one_hot(self):
        q_str = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        max_len = 5
        sym_dicts = [{'RY': 0, 'XY': 1, 'nop': 2, 'X': 3, 'Y': 4, 'ZZ': 5,
            'RX': 6, 'H': 7, 'XX': 8, 'nop': 9}, \
                     {'0': 0, '1': 1}, {'nop': 0, '1': 1, '0': 2, '3': 3}]
        unary_strs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0]]
        ref_de = [[7, 0, 0, 0.2], [3, 1, 0, 0.2], [6, 1, 2, 0.1], [8, 0, 3, 0.2], [1, 0, 1, 0.2]]
        ref_ohe = [[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0], [1, 0, 0, 0], 0.2],
                   [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1], [1, 0, 0, 0], 0.2],
                   [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 1], [0, 0, 1, 0], 0.1],
                   [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0], [0, 0, 0, 1], 0.2],
                   [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0], [0, 1, 0, 0], 0.2]]
        de, ohe = one_hot.to_one_hot(q_str, max_len, sym_dicts, unary_strs)
        for i in range(max_len):
            assert de[i] == ref_de[i]
            assert one_hot._compare_one_hot(ohe[i], ref_ohe[i])

    def test_from_one_hot(self):
        ohe = [[[0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 1, 0], [1, 0, 0, 0], 0.2],
                    [[0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1], [1, 0, 0, 0], 0.2],
                    [[0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1], [0, 0, 1, 0], 0.1],
                    [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0], [0, 0, 0, 1], 0.2],
                    [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0], [0, 1, 0, 0], 0.2]]
        reverse_dictionary_list = [{0:'RY', 1:'XY', 2:'nop', 3:'X', 4:'Y', 5:'ZZ', 6:'RX', 7:'H', 8:'XX'},
                             {0:'nop', 1:'0', 2:'1'}, {0:'nop', 1:'1', 2:'0', 3:'3'}]
        circuit_str = one_hot.from_one_hot(ohe, reverse_dictionary_list)
        ref_circuit_str = "H=0=nop=0.2@X=1=nop=0.2@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        assert circuit_str == ref_circuit_str

#    def test_ravel_list(self):
#        lst = [[[0,1],[1,2,3], []], [[1,2,3,4]]]
#        out_lst = utils.ravel_list(lst)
#        ref_lst = [0,1,1,2,3,1,2,3,4]
#        assert out_lst == ref_lst
#
#    def test_get_num_gates_topo(self):
#        circ_topo = [[[0], [1,2,3,5]], [[1,4], [2,5]]]
#        out_len = utils.get_num_gates_topo(circ_topo)
#        assert out_len == 6
#
#    def test_get_num_gates_qstring(self):
#        q_str = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@RY=0=1=0.2"
#        n_gates = utils.get_num_gates_qstring(q_str)
#        assert n_gates == 4
#
#
