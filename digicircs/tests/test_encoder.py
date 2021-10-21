import tequila as tq
from digicircs import encoder

class TestEncoder():
    def test_break_circuit(self):
        circuit = tq.gates.H(target=0) + tq.gates.X(target=1) + tq.gates.CRx(target=1,control=0,angle=0.1) + \
                  tq.gates.ExpPauli("X(0)X(3)",0.2) + tq.gates.ExpPauli("X(0)Y(1)",0.2)
        circuit_lst = encoder._break_circuit(circuit)
        ref_circuit_lst = [tq.gates.H(target=0), tq.gates.X(target=1), tq.gates.CRx(target=1,control=0,angle=0.1),
                           tq.gates.ExpPauli("X(0)X(3)",0.2), tq.gates.ExpPauli("X(0)Y(1)",0.2)]
        len_circuit = len(circuit_lst)
        for i in range(len_circuit):
            assert circuit_lst[i] == ref_circuit_lst[i].gates[0]

    def test_convert_gates_to_string(self):
        gate1 = tq.gates.H(target=0)
        gate2 = tq.gates.Rx(target=1, angle=0.1)
        gate3 = tq.gates.CRx(target=1, control=0, angle=0.1)
        gate4 = tq.gates.ExpPauli("X(0)X(3)", 0.2)

        str1 = "H=0=nop=nop"
        str2 = "RX=1=nop=0.1"
        str3 = "CRX=1=0=0.1"
        str4 = "XX=0=3=0.2"

        assert encoder._convert_gates_to_string(gate1.gates[0]) == str1
        assert encoder._convert_gates_to_string(gate2.gates[0]) == str2
        assert encoder._convert_gates_to_string(gate3.gates[0]) == str3
        assert encoder._convert_gates_to_string(gate4.gates[0]) == str4

    def test_encoder1(self):
        ''' no ExpPauli'''
        circuit = tq.gates.H(target=0) + tq.gates.X(target=1) \
                + tq.gates.CRx(target=1,control=0,angle=0.1)

        str_out = encoder.encoder(circuit)

        str_ref = "H=0=nop=nop@X=1=nop=nop@CRX=1=0=0.1"
        assert str_out == str_ref

    def test_encoder(self):
        circuit = tq.gates.H(target=0) + tq.gates.X(target=1) \
                + tq.gates.CRx(target=1,control=0,angle=0.1) \
                + tq.gates.ExpPauli("X(0)X(3)",0.2) + tq.gates.ExpPauli("X(0)Y(1)",0.2)

        str_out = encoder.encoder(circuit)

        str_ref = "H=0=nop=nop@X=1=nop=nop@CRX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        assert str_out == str_ref

        circuit = tq.gates.Rx(target=1, angle=0.4) + tq.gates.ExpPauli("X(0)Y(1)",angle=0.1)
        str_out = encoder.encoder(circuit)
        str_ref = "RX=1=nop=0.4@XY=0=1=0.1"
        assert str_out == str_ref



#if __name__ == "__main__":
#    obj = TestEncoder()
#    obj.test_break_circuit()
