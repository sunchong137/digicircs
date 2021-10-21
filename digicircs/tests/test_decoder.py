from digicircs import decoder
import unittest
import tequila as tq

class TestDecoder(unittest.TestCase):

    def test_gate_preprocess(self):
        str_i = "H=0=1=0.2"
        str_o = "H=0=nop=nop"
        assert decoder.gate_preprocess(str_i) == str_o

        str_i = "RX=0=1=nop"
        str_ref = "RX=0=nop=0.2"
        str_o = decoder.gate_preprocess(str_i)
        assert str_o.split("=")[:-1] == str_ref.split("=")[:-1]
        try:
            param = float(str_o.split("=")[-1])
        except:
            raise Exception("Failed to generate a random parameter!")

        str_i = "RX=0=1=nop"
        str_ref = "RX=0=nop=fd"
        str_o = decoder.gate_preprocess(str_i, fix_params=False)
        assert str_o.split("=")[:-1] == str_ref.split("=")[:-1]
        assert len(str_o.split("=")[-1]) == 4

        str_i = "CNOT=0=0=0.1"
        str_ref = "X=0=nop=nop"
        str_o = decoder.gate_preprocess(str_i)
        assert str_o == str_ref

        str_i = "CNOT=0=nop=0.1"
        str_ref = "X=0=nop=nop"
        str_o = decoder.gate_preprocess(str_i)
        assert str_o == str_ref

        str_i = "CRX=0=1=nop"
        str_ref = "CRX=0=1=0.2"
        str_o = decoder.gate_preprocess(str_i)
        assert str_o.split("=")[:-1] == str_ref.split("=")[:-1]
        try:
            param = float(str_o.split("=")[-1])
        except:
            raise Exception("Failed to generate a random parameter!")

        str_i = "CRX=0=1=nop"
        str_ref = "CRX=0=1=fd2e"
        str_o = decoder.gate_preprocess(str_i, fix_params=False)
        assert str_o.split("=")[:-1] == str_ref.split("=")[:-1]
        assert len(str_o.split("=")[-1]) == 4

        str_i = "W=0=1=nop" # not in the gates
        self.assertRaises(Exception, decoder.gate_preprocess, str_i)

    def test_qstring_preprocessing(self):
        str_i = "H=0=1=0.2@CNOT=0=0=0.1@CNOT=0=nop=0.1"
        str_ref = "H=0=nop=nop@X=0=nop=nop@X=0=nop=nop"
        str_o = decoder.qstring_preprocess(str_i)
        assert str_o == str_ref

    def test_convert_string_to_gates(self):
        str1 = "H=0=nop=nop"
        str2 = "RX=1=nop=0.1"
        str3 = "RX=1=0=0.1"
        str4 = "XX=0=3=0.2"
        str5 = "CRX=1=0=0.2"
        str6 = "CNOT=1=0=nop"
        gate1 = decoder.convert_string_to_gates(str1)
        gate2 = decoder.convert_string_to_gates(str2)
        gate3 = decoder.convert_string_to_gates(str3)
        gate4 = decoder.convert_string_to_gates(str4)
        gate5 = decoder.convert_string_to_gates(str5)
        gate6 = decoder.convert_string_to_gates(str6)

        assert gate1.gates[0] == tq.gates.H(target=0).gates[0]
        assert gate2.gates[0] == tq.gates.Rx(target=1,angle=0.1).gates[0]
        assert gate3.gates[0] == tq.gates.Rx(target=1,control=0,angle=0.1).gates[0]
        assert gate4.gates[0] == tq.gates.ExpPauli("X(0)X(3)",0.2).gates[0]
        assert gate5.gates[0] == tq.gates.CRx(target=1,control=0,angle=0.2).gates[0]
        assert gate6.gates[0] == tq.gates.CNOT(target=1,control=0).gates[0]
    # def test_convert_string_to_gates1(self):
    #     ''' single-qubit gate but control qubit is given'''
    #     str = "H=1=0=nop"
    #     gate = decoder.convert_string_to_gates(str)
    #     assert gate.gates[0] == tq.gates.H(target=1).gates[0]
    #
    # def test_convert_string_to_gates2(self):
    #     ''' two-qubit gate but control qubit = target qubit.'''
    #     str = "CRX=1=1=0.4"
    #     gate = decoder.convert_string_to_gates(str)
    #     assert gate.gates[0] == tq.gates.Rx(target=1, angle=0.4).gates[0]
    #
    # def test_convert_string_to_gates3(self):
    #     ''' two-qubit gate but control qubit is not given'''
    #     str = "CRX=1=nop=0.2"
    #     gate = decoder.convert_string_to_gates(str)
    #     assert gate.gates[0] == tq.gates.Rx(target=1, angle=0.2).gates[0]
    #
    # def test_convert_string_to_gates4(self):
    #     ''' static gate with a parameter.'''
    #     str = "H=1=nop=0.2"
    #     gate = decoder.convert_string_to_gates(str)
    #     assert gate.gates[0] == tq.gates.H(target=1).gates[0]
    #
    # def test_convert_string_to_gates5(self):
    #     '''parameters as variables'''
    #     str = "CRX=1=0=a"
    #     gate = decoder.convert_string_to_gates(str, fix_params=False)
    #     #assert gate.gates[0] == tq.gates.Rx(target=1, control=0, angle="a")
    #
    def test_decoder1(self):
        q_string = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1"
        ref_circ = tq.gates.H(target=0) + tq.gates.X(target=1) + tq.gates.Rx(target=1,angle=0.1)
        out_circ = decoder.decoder(q_string)
        n_gates = 3
        for i in range(n_gates):
            assert ref_circ.gates[i] == out_circ.gates[i]

    def test_decoder2(self):
        '''
        Test if control gates work.
        '''
        q_string = "CRX=0=1=0.1@CNOT=2=3=nop"
        out_circ = decoder.decoder(q_string)
        ref_circ = tq.gates.CRx(target=0, control=1, angle=0.1) + \
                   tq.gates.CNOT(target=2, control=3)
        n_gates = 2
        for i in range(n_gates):
            assert out_circ.gates[i] == ref_circ.gates[i]
    def test_decoder3(self):
        q_string = "RX=1=nop=0.4@XY=0=1=0.1"
        ref_circ = tq.gates.Rx(target=1, angle=0.4) + tq.gates.ExpPauli("X(0)Y(1)",angle=0.1)
        out_circ = decoder.decoder(q_string)
        assert ref_circ == out_circ
