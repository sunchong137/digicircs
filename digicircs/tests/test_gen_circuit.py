import unittest
from digicircs  import gen_circuit

class TestGenCircuit(unittest.TestCase):
    def test_gen_random_idx(self):
        # repeatedly generating 1000 numbers and see if they agree the weights
        weights = [0.2, 0.6, 0.2]
        n_rep = 1000
        n_0 = 0
        n_1 = 0
        n_2 = 0
        for i in range(n_rep):
            ind = gen_circuit._gen_random_idx(weights)
            if ind == 0:
                n_0 += 1
            elif ind == 1:
                n_1 += 1
            else:
                n_2 += 1
        # give a very coarse grained pass standard
        assert abs((n_0/n_rep) - weights[0]) < 0.05
        assert abs((n_1/n_rep) - weights[1]) < 0.05
        assert abs((n_2/n_rep) - weights[2]) < 0.05

    def test_random_elem_from_lst(self):
        lst = list(range(10))
        lst.remove(5)
        elem = gen_circuit._random_elem_from_lst(lst, 5, 3)
        assert elem in [2,3,4,6,7,8]
        assert len(lst) == 8

    def test_gen_circuit_topo_one_moment(self, layer: str = None, n_qubit: int = None):
        if layer == None:
            n_qubit = 10
            weights = [0.3, 0.5, 0.2]
            layer = gen_circuit._gen_circuit_topo_one_moment(n_qubit, weights = weights)

        # total number of qubit covered must be the number of qubits
        nq_layer = 0
        for i in range(len(layer)):
            nq_layer += len(layer[i])

        assert nq_layer <= n_qubit
        # no overlap among gates
        assert list(set(layer[0]).intersection(layer[1])) == []
        assert len(set(layer[0])) == len(layer[0])
        assert len(set(layer[1])) == len(layer[1])

    def test_gen_circuit_topo_one_moment2(self):

        n_qubit = 10
        weights = [0.3, 0.5, 0.2]
        layer = gen_circuit._gen_circuit_topo_one_moment(n_qubit, weights=weights, rand_seed=0)
        assert layer == [[5, 2, 4, 3, 9, 0, 6], []]

    def test_gen_circuit_topology(self):
        n_qubit = 6
        n_layer = 4
        weights = [0.2, 0.6, 0.2]
        q_circuit, a, b = gen_circuit.gen_circuit_topology(n_qubit, n_layer, weights)
        assert len(q_circuit) == n_layer
        for _layer in q_circuit:
            self.test_gen_circuit_topo_one_moment(_layer, n_qubit)


    def test_gen_circuit_topology2(self):
        n_qubit = 6
        n_layer = 4
        weights = [0.2, 0.6, 0.2]
        q_circuit, a, b = gen_circuit.gen_circuit_topology(n_qubit, n_layer, weights, rand_seed=0)
        assert q_circuit == [[[1, 2, 5, 0, 3], []], [[1, 2, 5, 0, 3], []], [[1, 2, 5, 0, 3], []], [[1, 2, 5, 0, 3], []]]
        assert a == 20
        assert b == 0

    def test_gen_gates_one_moment(self):
        layer = [[0,4], [2,3]]
        layer_str_out = gen_circuit.gen_gates_one_moment(layer, rand_seed = 0)
        layer_str_ref = "RZ=0=nop=nop@H=4=nop=nop@XZ=2=3=nop"
        assert layer_str_out == layer_str_ref

        layer = [[0,4], [2,3]]
        sgates_1q = []
        pgates_1q = ["RX"]
        sgates_2q = []
        pgates_2q = ["XY"]
        layer_str_out = gen_circuit.gen_gates_one_moment(layer,
                                                         sgates_1q=sgates_1q,
                                                         pgates_1q=pgates_1q,
                                                         sgates_2q=sgates_2q,
                                                         pgates_2q=pgates_2q,
                                                         rand_seed=0)
        assert layer_str_out == "RX=0=nop=nop@RX=4=nop=nop@XY=2=3=nop"

    def test_gen_circuit_gates(self):
        topo_lst = [[[0,4],[2,3]], [[1,5],[0,2,3,4]]]
        topo_str_out = gen_circuit.gen_circuit_gates(topo_lst, rand_seed=0)
        topo_str_ref = "RZ=0=nop=nop@H=4=nop=nop@XZ=2=3=nop@RZ=1=nop=nop@H=5=nop=nop@XZ=0=2=nop@CNOT=3=4=nop"
        assert topo_str_out == topo_str_ref

    def test_gen_circuit_gates2(self):
        topo_str_out = gen_circuit.gen_circuit_gates(rand_seed=0, n_qubit=4, n_moments=2)
        assert topo_str_out == "RZ=3=nop=nop@XZ=1=0=nop@RZ=3=nop=nop@XZ=1=0=nop"

    def test_add_params(self):
        q_string = "RZ=0=nop=nop@RZ=4=nop=nop@CNOT=2=3=nop"
        out_string, n_param = gen_circuit.add_params(q_string, rand_seed=0, return_nparam=True)
        params = [0.09762701, 0.43037873, 0.20552675]
        ref_string = "RZ=0=nop=2.9563@RZ=4=nop=1.8851@CNOT=2=3=nop"
        assert out_string == ref_string
        assert n_param == 2

    def test_add_params1(self):
        q_string = "RZ=0=nop=nop@RZ=4=nop=nop@CNOT=2=3=nop"
        params = [0.09762701, 0.43037873, 0.20552675]
        out_string, n_param = gen_circuit.add_params(q_string, params=params, return_nparam=True)
        ref_string = "RZ=0=nop=0.0976@RZ=4=nop=0.4304@CNOT=2=3=nop"
        assert out_string == ref_string

    def test_add_params2(self):
        q_string = "RZ=0=nop=nop@RZ=4=nop=nop@CNOT=2=3=nop"
        out_string, n_param = gen_circuit.add_params(q_string, fix_params=False, return_nparam=True)
        ref_string = "RZ=0=nop=nop0@RZ=4=nop=nop1@CNOT=2=3=nop"
        assert out_string == ref_string

    def test_circuit_from_scratch1(self):
        """the number of gates is given"""
        n_qubit = 4
        n_gates = 6
        out_q_str, n_p = gen_circuit.circuit_from_scratch(n_qubit,
                                                          n_gates,
                                                          rand_seed=0,
                                                          fix_params=True)

        ref_q_str = "ZZ=3=1=3.9175@XY=3=1=3.5265@CNOT=1=0=nop@RZ=3=nop=2.9563@RZ=3=nop=1.8851@X=2=nop=nop"
        assert n_p == 4
        assert out_q_str == ref_q_str

    def test_circuit_from_scratch2(self):
        """The range of the number of gates is given"""
        n_qubit = 6
        min_ngates = 3
        max_ngates = 5
        out_q_str, n_p = gen_circuit.circuit_from_scratch(n_qubit,
                                                          min_ngates=min_ngates,
                                                          max_ngates=max_ngates,
                                                          rand_seed=0,
                                                          fix_params=True)

        ref_q_str = "RZ=3=nop=2.4526@ZZ=3=4=1.6450@YY=0=3=1.8879"
        assert n_p == 3
        assert out_q_str == ref_q_str

    def test_circuit_from_scratch3(self):
        """fix_params is False"""
        n_qubit = 4
        n_gates = 6
        out_q_str, n_p = gen_circuit.circuit_from_scratch(n_qubit, n_gates,
                                                          rand_seed=0,
                                                          fix_params=False)

        ref_q_str = "ZZ=3=1=param2@XY=3=1=param3@CNOT=1=0=nop@RZ=3=nop=param0@RZ=3=nop=param1@X=2=nop=nop"
        assert n_p == 4
        assert out_q_str == ref_q_str
