from digicircs import multi_hot
import torch
import numpy

class TestMultiHot():
    def test_to_multi_hot(self):
        q_str = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        max_len = 5
        sym_dicts = [{'RY': 0, 'XY': 1, 'nop': 2, 'X': 3, 'Y': 4, 'ZZ': 5,
            'RX': 6, 'H': 7, 'XX': 8, 'nop': 9}, \
                     {'0': 0, '1': 1}, {'nop': 0, '1': 1, '0': 2, '3': 3}]
        unary_strs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0]]
        ref_de = [[7, 0, 0, 0.2], [3, 1, 0, 0.2], [6, 1, 2, 0.1], [8, 0, 3, 0.2], [1, 0, 1, 0.2]]
        ref_mhe = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0.2],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0.2],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0.1],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0.2],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.2]]
        de, mhe = multi_hot.to_multi_hot(q_str, max_len, sym_dicts, unary_strs)
        for i in range(max_len):
            #assert set(de[i]) == set(ref_de[i])
            assert de[i] == ref_de[i]
            assert ref_mhe[i] == mhe[i]
            #assert one_hot._compare_one_hot(ohe[i], ref_ohe[i])

    def test_to_multi_hot1(self):
        q_str = "H=0=nop=nop@X=1=nop=nop@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        max_len = 5
        sym_dicts = [{'RY': 0, 'XY': 1, 'nop': 2, 'X': 3, 'Y': 4, 'ZZ': 5,
            'RX': 6, 'H': 7, 'XX': 8}, \
                     {'0': 0, '1': 1}, {'nop': 0, '1': 1, '0': 2, '3': 3}]
        unary_strs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0]]
        ref_de = [[7, 0, 0], [3, 1, 0], [6, 1, 2], [8, 0, 3], [1, 0, 1]]
        ref_mhe = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]]
        de, mhe = multi_hot.to_multi_hot(q_str, max_len, sym_dicts,
                                         unary_strs, encode_params=False)
        for i in range(max_len):
            #assert set(de[i]) == set(ref_de[i])
            assert de[i] == ref_de[i]
            assert ref_mhe[i] == mhe[i]
            #assert one_hot._compare_one_hot(ohe[i], ref_ohe[i])

    def test_to_multi_hot2(self):
        # TODO:
        q_str = "nop=0=nop=nop@nop=1=nop=nop@nop=1=nop=nop@nop=0=nop=nop"
        max_len = 4
        sym_dicts = [{'RY': 0, 'XY': 1, 'nop': 2, 'X': 3, 'Y': 4, 'ZZ': 5,
            'RX': 6, 'H': 7, 'XX': 8}, \
                     {'0': 0, '1': 1}, {'nop': 0, '1': 1, '0': 2, '3': 3}]
        unary_strs = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0], [0, 0, 0, 0]]
        ref_mhe = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]]
        de, mhe = multi_hot.to_multi_hot(q_str, max_len, sym_dicts,
                                         unary_strs, encode_params=False)
        for i in range(max_len):
            assert ref_mhe[i] == mhe[i]

    def test_from_multi_hot(self):
        mhe =      [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0.2],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0.2],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0.1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0.2],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.2]]
        reverse_dictionary_list = [{0:'RY', 1:'XY', 2:'nop', 3:'X', 4:'Y', 5:'ZZ', 6:'RX', 7:'H', 8:'XX'},
                             {0:'nop', 1:'0', 2:'1'}, {0:'nop', 1:'1', 2:'0', 3:'3'}]
        circuit_str = multi_hot.from_multi_hot(mhe, reverse_dictionary_list)
        ref_circuit_str = "H=0=nop=0.2@X=1=nop=0.2@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        assert circuit_str == ref_circuit_str

    def test_from_multi_hot1(self):
        mhe =      [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]]
        reverse_dictionary_list = [{0:'RY', 1:'XY', 2:'nop', 3:'X', 4:'Y', 5:'ZZ', 6:'RX', 7:'H', 8:'XX'},
                             {0:'nop', 1:'0', 2:'1'}, {0:'nop', 1:'1', 2:'0', 3:'3'}]
        circuit_str = multi_hot.from_multi_hot(mhe, reverse_dictionary_list, encode_params=False)

        ref_circuit_str = "H=0=nop=nop0@X=1=nop=nop1@RX=1=0=nop2@XX=0=3=nop3@XY=0=1=nop4"
        assert circuit_str == ref_circuit_str


    def test_from_multi_hot_noisy(self):
        mhe = numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0.2],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0.2],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0.1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0.2],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.2]])
        mhe_noisy = multi_hot.add_noise_to_mhe(mhe, upper_bound = 0.95)
        reverse_dictionary_list = [{0:'RY', 1:'XY', 2:'nop', 3:'X', 4:'Y', 5:'ZZ', 6:'RX', 7:'H', 8:'XX'},
                             {0:'nop', 1:'0', 2:'1'}, {0:'nop', 1:'1', 2:'0', 3:'3'}]
        circuit_str = multi_hot.from_multi_hot(mhe_noisy, reverse_dictionary_list)
        ref_circuit_str = "H=0=nop=0.2@X=1=nop=0.2@RX=1=0=0.1@XX=0=3=0.2@XY=0=1=0.2"
        assert circuit_str == ref_circuit_str

    def test_add_noise_to_mhe_numpy(self):

        mhe = numpy.array([[0, 0, 1, 0, 0, 0, 0, 1.2],
                           [0, 0, 0, 1, 0, 0, 1, 0.1],
                           [0, 1, 0, 0, 0, 0, 0, 0.3]])
        mhe_out = multi_hot.add_noise_to_mhe(mhe, upper_bound = 0.95, rand_seed = 0)
        mhe_ref = numpy.array([[0.52137283, 0.6794299, 1., 0.51763902, 0.40247206, 0.61359941,
                                0.41570785, 1.2],
                               [0.91547962, 0.36426944, 0.75213879, 1., 0.53964233, 0.87931681,
                                1., 0.1],
                               [0.01920748, 1., 0.73924891, 0.82651154, 0.92968743, 0.75920064,
                                0.43840539, 0.3]])
        assert numpy.linalg.norm(mhe_out - mhe_ref) < 1e-6

    def test_add_noise_to_mhe_numpy1(self):

        mhe = numpy.array([[0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 1],
                           [0, 1, 0, 0, 0, 0, 0]])
        mhe_out = multi_hot.add_noise_to_mhe(mhe, upper_bound=0.95, rand_seed=0,
                                             encode_params=False)
        mhe_ref = numpy.array([[0.52137283, 0.6794299, 1., 0.51763902, 0.40247206,
                                0.61359941, 0.41570785],
                               [0.84718435, 0.91547962, 0.36426944, 1., 0.50245017,
                                0.53964233, 1.        ],
                               [0.06748426, 1., 0.01920748, 0.79098885, 0.73924891,
                                0.82651154,0.92968743]])

        assert numpy.linalg.norm(mhe_out - mhe_ref) < 1e-6

    def test_add_noise_to_mhe_torch(self):

        mhe = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0.2],
                           [0, 0, 0, 1, 0, 0, 1, 0.1],
                           [0, 1, 0, 0, 0, 0, 0, 0.3]])
        mhe_out = multi_hot.add_noise_to_mhe(mhe, upper_bound = 0.95, rand_seed = 0)
        #torch.set_printoptions(precision=8)
        mhe_ref = torch.tensor([[0.47144374, 0.72981071, 1.00000000, 0.12542896, 0.29205167, 0.60237473,
                                 0.46558875, 0.20000000],
                                [0.43284658, 0.60069096, 0.33144879, 1.00000000, 0.02120947, 0.16041599,
                                 1.00000000, 0.10000000],
                                [0.66278422, 1.00000000, 0.15297799, 0.26815516, 0.64752811, 0.86943424,
                                 0.37724492, 0.30000001]])

        assert torch.linalg.norm(mhe_out - mhe_ref) < 1e-6

    def test_remove_noise_mhe_numpy(self):
        mhe = numpy.array([[0.47144374, 0.72981071, 1.00000000, 0.12542896, 0.29205167, 0.60237473,
                             0.46558875, 0.20000000],
                            [0.43284658, 0.60069096, 0.33144879, 1.00000000, 0.02120947, 0.16041599,
                             1.00000000, 0.10000000],
                            [0.66278422, 1.00000000, 0.15297799, 0.26815516, 0.64752811, 0.86943424,
                             0.37724492, 0.30000001]])
        mhe_out = multi_hot.remove_noise_mhe(mhe)
        mhe_ref = numpy.array([[0, 0, 1, 0, 0, 0, 0, 0.2],
                                [0, 0, 0, 1, 0, 0, 1, 0.1],
                                [0, 1, 0, 0, 0, 0, 0, 0.3]])
        assert numpy.linalg.norm(mhe_out - mhe_ref) < 1e-6

    def test_remove_noise_mhe_torch(self):
        mhe = torch.tensor([[0.47144374, 0.72981071, 1.00000000, 0.12542896, 0.29205167, 0.60237473,
                             0.46558875, 0.20000000],
                            [0.43284658, 0.60069096, 0.33144879, 1.00000000, 0.02120947, 0.16041599,
                             1.00000000, 0.10000000],
                            [0.66278422, 1.00000000, 0.15297799, 0.26815516, 0.64752811, 0.86943424,
                             0.37724492, 0.30000001]])
        mhe_out = multi_hot.remove_noise_mhe(mhe)
        mhe_ref = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0.2],
                                [0, 0, 0, 1, 0, 0, 1, 0.1],
                                [0, 1, 0, 0, 0, 0, 0, 0.3]])
        assert torch.linalg.norm(mhe_out - mhe_ref) < 1e-6
