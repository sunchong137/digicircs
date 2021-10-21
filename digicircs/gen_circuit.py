r'''
Random circuits are generated for the training purpose. Two ways are provided:

1. Hierarchical generation.
       circuit topologies -> specific gates -> add parameters to gates

2. A completely random circuit without fixed number of moments or gates.

Authors:
    Chong Sun <sunchong137@gmail.com>
    Hannah Sim <hsim13372@gmail.com>
'''
import numpy
import random
import warnings
import copy
from digicircs import __config__
from digicircs.utils import misc

# Default gates (Static and Parameterized)
DEFAULT_GATES = __config__._default_gates
SGATES_1Q = DEFAULT_GATES["_static_gates_for_1qubit"]
SGATES_2Q = DEFAULT_GATES["_static_gates_for_2qubits"]
PGATES_1Q = DEFAULT_GATES["_parameterized_gates_for_1qubit"]
PGATES_2Q = DEFAULT_GATES["_parameterized_gates_for_2qubit"]
GATES_1Q = SGATES_1Q + PGATES_1Q
GATES_2Q = PGATES_2Q + SGATES_2Q

def gen_circuit_topology(n_qubit: int, n_moments: int,
                         weights: list=[0.2, 0.6, 0.2],
                         local_rot_moment: bool=False, max_dist: int=None,
                         rand_seed: int=None, **kwargs):
    '''
    Generate an arbitrary but valid circuit topology for given number
    of qubits and moments. For each moment, two types of gates are included:

        - 1-qubit gates  (identity not included)
        - 2-qubit gates

    Args:
        :n_qubit: number of qubits in the circuit
        :n_moments: number of moments in the circuit
    Kwargs:
        :weights: weights to generate the three types of gates
        :local_rot_layer: whether to have an initial moment of local rotations.
        :max_dist: maximun distance between target and control qubits.
    Returns:
        :list: A list of moments
        :n_1q: Number of 1-qubit gates
        :n_2q: Number of 2-qubit gates
    Examples:
        >>> n_qubit, n_moments = 4, 2
        >>> q_topo = gen_circuit_topology(n_qubit, n_moments)
        >>> print(q_topo)
            [[[0], [2, 3]], [[], [0, 3, 2, 1]]]
    '''
    n_1q, n_2q = 0, 0
    circuit_moments = []

    if local_rot_moment:
    	moment = [[],[]]
    	moment[0] = [q for q in range(n_qubit)]
    	n_1q += n_qubit
    	circuit_moments.append(moment)
    	n_moments -= 1

    for i in range(n_moments):
        moment = _gen_circuit_topo_one_moment(n_qubit, weights, max_dist=max_dist, rand_seed=rand_seed)
        circuit_moments.append(moment)
        n_1q += len(moment[0])
        n_2q += len(moment[1])//2

    return circuit_moments, n_1q, n_2q


def _gen_circuit_topo_one_moment(n_qubit: int, weights: list=[0.2, 0.6, 0.2],
                                 max_dist: int=None, rand_seed:int=None, **kwargs):
    '''
    Generate one moment of gates for given number of qubits.

    Args:
        :n_qubit: number of qubits in the circuit.
    Kwargs:
        :weights: weights to generate the three types of gates: identity, 1-qubit gates and 2-qubit gates.
        :max_dist: the maximum distance for 2-qubit gates.
    Returns:
        :list: A list representing one moment of gates with the following format:

        [[indices of qubits acted by 1-q gates],
         [pairs of target and control qubits acted by 2-q gates]]

         Note that the order of the last list is
         [target1, control1, target2, control2, ..]
    '''

    if max_dist == None:
        max_dist = n_qubit
    qubit_lst = list(range(n_qubit))
    w_gated = (weights[1] + 2*weights[2]) / ((weights[0] + weights[1] + 2*weights[2]))
    n_gated = int(n_qubit * w_gated)

    random.seed(rand_seed)
    qubit_lst = random.sample(qubit_lst, n_gated)
    moment = [[],[]]

    while(qubit_lst > []):
        qubit_targ = _random_elem_from_lst(lst=qubit_lst, rand_seed=rand_seed)
        gate = _gen_random_idx(weights[1:], rand_seed=rand_seed) + 1# gate is 0, 1, 2
        if gate == 2:
            if qubit_lst == []: # only 1 qubit left
                gate = 1
            else:
                qubit_ctrl = _random_elem_from_lst(lst=qubit_lst,
                                                   qubit_targ=qubit_targ,
                                                   max_dist=max_dist,
                                                   rand_seed=rand_seed)
                moment[1].append(qubit_targ)
                moment[1].append(qubit_ctrl)
        if gate == 1: # do not use elif or else because above the gate could be changed
            moment[0].append(qubit_targ)

    return moment

def gen_circuit_gates(topo_lst: list=None, gate_pool: dict=None,
                      rand_seed: int=None, n_qubit: int=None,
                      n_moments: int=None, weights: list=[0.2, 0.4, 0.4],
                      local_rot_moment: bool=False,
                      max_dist: int=None, **kwargs):
    '''
    Fill the gates randomly given a certain multi-moment circuit topology.

    Kwargs:
        :topo_lst: A list representing the topologies of moments.
        :gate_pool: Dictionary of lists:
            - sgates_1q: list of symbols of static 1-qubit gates.
            - pgates_1q: list of symbols of parametrized 1-qubit gates.
            - sgates_2q: list of symbols of static 2-qubit gates.
            - pgates_2q: list of symbols of parametrized 2-qubit gates.
        :rand_seed: the seed for random generator, do not give it value otherwise not random.
        :n_qubit: number of qubits in the circuit
        :n_moments: number of moments in the circuit
    Returns:
        :str: A string representing the gates in the circuit.
    Examples:
        >>> topo_lst = [[[0,4],[2,3]], [[1,5],[0,2,3,4]]]
        >>> q_str_gates = gen_circuit_gates(topo_lst)
        >>> print(q_str_gates)
            Rz=0=nop=nop@Rz=4=nop=nop@CNOT=2=3=nop@Rz=1=nop=nop@Rz=5=nop=nop@CNOT=0=2=nop@CNOT=3=4=nop
    '''
    if topo_lst is None:
        assert n_qubit is not None and n_moments is not None, \
        "Circuit topology cannot be constructed without specifying n_qubit and n_moments!"
        topo_lst = gen_circuit_topology(n_qubit=n_qubit, n_moments=n_moments,
                                        weights=weights,
                                        local_rot_moment=local_rot_moment,
                                        max_dist=max_dist, rand_seed=rand_seed)[0]

    try:
        sgates_1q = gate_pool["sgates_1q"]
    except:
        sgates_1q = SGATES_1Q
    try:
        pgates_1q = gate_pool["pgates_1q"]
    except:
        pgates_1q = PGATES_1Q
    try:
        sgates_2q = gate_pool["sgates_2q"]
    except:
        sgates_2q = SGATES_2Q
    try:
        pgates_2q = gate_pool["pgates_2q"]
    except:
        pgates_2q = PGATES_2Q

    # start generating circuits
    q_string = ""

    for moment_lst in topo_lst:
        moment_str = gen_gates_one_moment(moment_lst, sgates_1q=sgates_1q,
                                 pgates_1q=pgates_1q, sgates_2q=sgates_2q,
                                 pgates_2q=pgates_2q, rand_seed=rand_seed)

        q_string += moment_str
        q_string += "@"

    return q_string[:-1]

def gen_gates_one_moment(topo_lst: list, sgates_1q: list=None,
                         pgates_1q: list=None, sgates_2q: list=None,
                         pgates_2q: list=None, rand_seed: int=None, **kwargs):
    '''
    Fill the gates randomly given a certain single-moment circuit topology.
    current list of gates supported:

        `1-qubit: "X", "Y", "Z", "H", "Rx", "Ry", "Rz"`,

        `2-qubit: "CRx", "CRy", "CRz", "CNOT"`,

    .. todo:: Chong: Check if we want to use all of the gates (include ExpPauli gates)

    Args:
        :topo_lst: A list representing the topology of the moment.
    Kwargs:
        :sgates_1q: list of symbols of static 1-qubit gates.
        :pgates_1q: list of symbols of parametrized 1-qubit gates.
        :sgates_2q: list of symbols of static 2-qubit gates.
        :pgates_2q: list of symbols of parametrized 2-qubit gates.
        :rand_seed: the seed for random generator, do not give it value otherwise not random.
    Returns:
        :str: A string representing the gates in this circuit moment.
    Examples:
        >>> topo_lst = [[0], [2, 1]]
        >>> moment_str = gen_gates_one_moment(topo_lst)
        >>> print(moment_str)
            X=0=nop=nop@CRZ=2=1=nop
    '''
    if sgates_1q is None:
        sgates_1q = SGATES_1Q
    if pgates_1q is None:
        pgates_1q = PGATES_1Q
    if sgates_2q is None:
        sgates_2q = SGATES_2Q
    if pgates_2q is None:
        pgates_2q = PGATES_2Q

    sym_gates_1q = sgates_1q + pgates_1q
    sym_gates_2q = sgates_2q + pgates_2q

    moment_str = ""
    # 1-qubit gates
    random.seed(rand_seed)
    for site in topo_lst[0]:
        gate = random.choice(sym_gates_1q)
        gate_str = gate + "=" + str(site) + "=nop=nop@"
        moment_str += gate_str
    # 2-qubit gates
    assert len(topo_lst[1]) % 2 == 0
    n_2q = len(topo_lst[1]) // 2
    for i in range(n_2q):
        targ = 2 * i
        ctrl = 2 * i + 1
        gate =  random.choice(sym_gates_2q)
        gate_str = gate + "=" + str(topo_lst[1][targ]) + "=" + str(topo_lst[1][ctrl]) + "=nop@"
        moment_str += gate_str

    return moment_str[:-1]

def gen_circuit_gates_fixed_n_params(topo_lst: list, ngates_1q2q: list, n_params: int,
                                     weights_1q2q: list = [0.5, 0.5],
                                     sgates_1q: list=None, sgates_2q: list=None,
                                     pgates_1q: list=None, pgates_2q: list=None,
                                     strategy: str='random',
                                     rand_seed: int=None,
                                     local_rot_moment: bool=False,
                                     **kwargs):
    r'''
    Fill the gates randomly given a certain multi-moment circuit topology
    and a fixed number of parameters (and corresponding gates) to allocate.

    Args:
        :topo_lst: A list representing the topologies of moments.
        :ngates_1q2q: A list of numbers of 1-qubit and 2-qubit gates.
        :n_params: Number of parameterized gates to add.
    Kwargs:
        :weights_1q2q: list of weights for allocating parameterized 1q or 2q gates.
        :sgates_1q: list of symbols of static 1-qubit gates.
        :sgates_2q: list of symbols of static 2-qubit gates.
        :pgates_1q: list of symbols of parameterized 1-qubit gates.
        :pgates_2q: list of symbols of parameterized 2-qubit gates.
        :strategy: string indicating strategy for allocating parameterized gates. Supported values: 'random', 'early'

                        -'random' : randomly distribute parameterized gates.

                        -'early' :  distribute parameterized gates from start of circuit.

        :rand_seed: the seed for random generator, do not give it value otherwise not random.
        :local_rot_layer: whether to have an initial moment of local rotations
    Returns:
        :str: A string representing the gates in the circuit.
    Examples:
        >>> topo_lst = [[[0,4],[2,3]], [[1,5],[0,2,3,4]]]
        >>> q_str_gates = gen_circuit_gates(topo_lst)
        >>> print(q_str_gates)
            Rz=0=nop=nop@Rz=4=nop=nop@CNOT=2=3=nop@Rz=1=nop=nop@Rz=5=nop=nop@CNOT=0=2=nop@CNOT=3=4=nop
    '''
    if n_params > numpy.sum(ngates_1q2q):
        n_params = numpy.sum(ngates_1q2q)
        warnings.warn("Number of parameterized gates must be <= number of gates in topology.")
        #raise ValueError('Number of parameterized gates must be <= number of gates in topology.')
    if sgates_1q is None:
        sgates_1q = SGATES_1Q
    if sgates_2q is None:
        sgates_2q = SGATES_2Q
    if pgates_1q is None:
        pgates_1q = PGATES_1Q
    if pgates_2q is None:
        pgates_2q = PGATES_2Q

    # Numbers of parameterized gates to allocate
    n_p1q = int(weights_1q2q[0]*n_params)
    n_p2q = n_params - n_p1q

    # Check if allocation is possible
    if n_p1q >= ngates_1q2q[0]:
        n_p1q = ngates_1q2q[0]
        n_p2q = n_params - n_p1q
        warnings.warn('''Input weights of 1q and 2q gates were not possible.
                      Setting n_p1q to {0} and n_p2q to {1}...'''.format(n_p1q, n_p2q))
    elif n_p2q >= ngates_1q2q[1]:
        n_p2q = ngates_1q2q[1]
        n_p1q = n_params - n_p2q
        warnings.warn('''Input weights of 1q and 2q gates were not possible.
                      Setting n_p1q to {0} and n_p2q to {1}...'''.format(n_p1q, n_p2q))

    q_string = ""
    if strategy == 'early':
        ordering = numpy.arange(len(topo_lst)).astype(int)
    elif strategy == 'random':
        ordering = numpy.arange(len(topo_lst)).astype(int)
        numpy.random.shuffle(ordering)
    elif strategy == 'late':
        ordering = numpy.arange(len(topo_lst))[::-1].astype(int)
    else:
        raise ValueError('Invalid strategy. Choices are: early, random, and late.')

    # Add to moments
    moment_strings = []
    for moment_lst in numpy.array(topo_lst)[ordering]:

        # Number of 1q, 2q gate slots at current circuit moment
        n1q, n2q = len(moment_lst[0]), len(moment_lst[1])//2

        if n_p1q - n1q < 0:
            n1q = n_p1q
        if n_p2q - n2q < 0:
            n2q = n_p2q

        moment_str = gen_gates_one_moment_fixed_n_params(n1q, n2q,
                                                         moment_lst,
                                                         sgates_1q = sgates_1q,
                                                         sgates_2q = sgates_2q,
                                                         pgates_1q = pgates_1q,
                                                         pgates_2q = pgates_2q,
                                                         rand_seed = rand_seed,
                                                         local_rot_moment=local_rot_moment)
        moment_strings.append(moment_str)

        # Decrement number of parameterized gates to allocate
        n_p1q -= n1q
        n_p2q -= n2q

    for moment_str in numpy.array(moment_strings)[ordering]:
        if moment_str == "":
            warnings.warn("An empty layer is generated!")
        else:
            q_string += moment_str
            q_string += "@"

    return q_string[:-1]

def gen_gates_one_moment_fixed_n_params(n_params_1q: int,
                                        n_params_2q: int,
                                        topo_lst: list,
                                        sgates_1q: list = None, sgates_2q: list = None,
                                        pgates_1q: list = None, pgates_2q: list = None,
                                        rand_seed: int = None,
                                        local_rot_moment: bool = False,
                                        **kwargs):
    '''
    Fill the gates randomly given a certain single-moment circuit topology.
    current list of gates supported:

        `1-qubit: "X", "Y", "Z", "H", "Rx", "Ry", "Rz"`,

        `2-qubit: "CRX", "CRY", "CRZ", "CNOT"`,

    Args:
        :n_params_1q: Number of 1q gates to parameterize.
        :n_params_2q: Number of 2q gates to parameterize.
        :topo_lst: A list representing the topology of the moment.
    Kwargs:
        :sgates_1q: list of symbols of static 1-qubit gates.
        :sgates_2q: list of symbols of static 2-qubit gates.
        :pgates_1q: list of symbols of parameterized 1-qubit gates.
        :pgates_2q: list of symbols of parameterized 2-qubit gates.
        :rand_seed: the seed for random generator, do not give it value otherwise not random.
        :local_rot_layer: whether to have an initial moment of local rotations
    Returns:
        :str: A string representing the gates in this circuit moment.
    '''
    if sgates_1q is None:
        sgates_1q = SGATES_1Q
    if sgates_2q is None:
        sgates_2q = SGATES_2Q
    if pgates_1q is None:
        pgates_1q = PGATES_1Q
    if pgates_2q is None:
        pgates_2q = PGATES_2Q

    if local_rot_moment:
    	try:
    		sgates_1q.remove('Z')
    	except:
    		pass
    	try:
    		pgates_1q.remove('RZ')
    	except:
    		pass

    moment_str = ""
    # 1-qubit gates
    for site in topo_lst[0]:

        random.seed(rand_seed)
        if n_params_1q > 0: # still need to allocate
            gate = random.choice(pgates_1q)
        else: # done allocating parameterized gates
            gate = random.choice(sgates_1q)
        gate_str = gate + "=" + str(site) + "=nop=nop@"
        moment_str += gate_str
        n_params_1q -= 1

    # 2-qubit gates
    assert len(topo_lst[1]) % 2 == 0
    n_2q = len(topo_lst[1]) // 2
    for i in range(n_2q):
        targ = 2 * i
        ctrl = 2 * i + 1
        random.seed(rand_seed)
        if n_params_2q > 0:
            gate = random.choice(pgates_2q)
        else:
            gate = random.choice(sgates_2q)
        gate_str = gate + "=" + str(topo_lst[1][targ]) + "=" + str(topo_lst[1][ctrl]) + "=nop@"
        moment_str += gate_str
        n_params_2q -= 1

    return moment_str[:-1]

def add_params(q_string: str, pgates_1q: list = None,
               pgates_2q: list = None, rand_seed: int = None,
               params: list = None, fix_params: bool = True,
               return_nparam: bool = False, **kwargs):
    '''
    Add random parameters to the gates.

    Args:
        :q_string: the string that stores the circuit information.
    Kwargs:
        :pgates_1q: list of symbols of parameterized 1-qubit gates.
        :pgates_2q: list of symbols of parameterized 2-qubit gates.
        :rand_seed: seed for generating random parameters, if given, then the distribution is fixed.
        :params: list of pre-computed parameters, default is None.
        :fix_params: if True, the parameters are fixed as numbers, otherwise as variables.
        :return_nparam: return the number of parameters.
    Returns:
        :str: A string of the same circuit but with random parameters for each gates.
        :int: number of parameters.
    Examples:
        >>> q_string = "X=0=nop=nop@CRZ=2=1=nop"
        >>> q_string_w_param, n_param = add_params(q_string)
        >>> print(q_string_w_param)
            X=0=nop=nop@CRZ=2=1=2.304
        >>> print(n_param)
            1

    '''
    if pgates_1q is None:
        pgates_1q = PGATES_1Q
    if pgates_2q is None:
        pgates_2q = PGATES_2Q
    pgates = pgates_1q + pgates_2q
    pgates = [x.lower() for x in pgates]

    # TODO: check about the range of parameters.
    gates = list(q_string.split("@"))
    n_gate = len(gates)

    if fix_params and params is None:
        params = misc.random_array(n_gate, distrib = "normal", rand_seed = rand_seed)

    q_string_w_param = ""
    ct = 0 # counter for parameters
    for i in range(n_gate):
        # Only add parameter values to parameterized gates
        gate_rep = list(gates[i].split("="))
        if gate_rep[0].lower() in pgates:
            if fix_params:
                gates[i] = gates[i][:-3] # remove nop
                gates[i] += "%1.4f"%params[ct]
            else:
                gates[i] += "%d"%ct #make different strings
            ct += 1

        q_string_w_param += gates[i]
        q_string_w_param += "@"
    if return_nparam:
        n_params = ct
        return q_string_w_param[:-1], n_params
    else:
        return q_string_w_param[:-1]

def circuit_from_scratch(n_qubit: int, n_gates: int=None, min_ngates: int=5,
                         max_ngates: int=100, weights: list=[0.5, 0.5],
                         max_dist: int=None, rand_seed: int=None,
                         fix_params: bool=True, **kwargs):
    '''
    Generate a totally random circuit from scratch given the number of qubits.

    Args:
        :n_qubit: number of qubits in the circuit.
    Kwargs:
        :n_gates: number of gates. If not specified, generate a number between [min_gates, max_gates]
        :min_ngates: minimum number of gates to generate.
        :max_ngates: maximum number of gates to generate.
        :weights: #1q_gates : #2q_gates
        :max_dist: maximun distance between target and control qubits.
        :rand_seed: random generator seed, used for test, do not assign value!
        :fix_params: if True, the generate a specific number for the parameters.
    Returns:
        :str: a string containing the gates with order.
        :num_params: number of parameters
    Examples:
        >>> n_qubit = 4
        >>> q_str = circuit_from_scratch(n_qubit, min_ngates = 2, max_ngates=10)
        >>> print(q_str)
            X=0=nop=0.1@H=2=nop=0.4@CRX=0=1=0.5=CRZ=0=3=0.8
    '''
    # for test-only
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    if n_gates is None:
        n_gates = numpy.random.randint(min_ngates, max_ngates)
    # normalize weights
    tot_weight = weights[0] + weights[1]
    if abs(tot_weight - 1) > 1e-10:
        weights[0] /= tot_weight
        weights[1] /= tot_weight

    n_1q_gates = int(n_gates * weights[0])
    n_2q_gates = n_gates - n_1q_gates
    qubit_lst = list(range(n_qubit))

    gate_strs = []

    if fix_params:
        params_1q = numpy.random.normal(numpy.pi/2., numpy.pi/4., n_1q_gates)
        params_2q = numpy.random.normal(numpy.pi/2., numpy.pi/3., n_2q_gates)

    num_params = 0
    for i in range(n_1q_gates):
        _gate = random.choice(GATES_1Q)
        _targ = random.choice(qubit_lst)
        if _gate in PGATES_1Q:
            if fix_params:
                gstr = _gate + "=" + str(_targ) + "=nop=" + "{:1.4f}".format(params_1q[i])
            else:
                gstr = _gate + "=" + str(_targ) + "=nop=" + "param{}".format(num_params)
            num_params += 1
        else:
            gstr = _gate + "=" + str(_targ) + "=nop=nop"

        gate_strs.append(gstr)

    for i in range(n_2q_gates):
        _gate = random.choice(GATES_2Q)
        _targ = random.choice(qubit_lst)

        # get ctrl qubit
        if max_dist is None:
            min_idx = 0
            max_idx = n_qubit - 2 # remove the target qubit
        else:
            min_idx = min(_targ - max_dist, 0)
            max_idx = max(_targ + max_dist, n_qubit - 1) - 1 # removed this qubit
        temp_lst = copy.copy(qubit_lst)
        temp_lst.remove(_targ)
        _ctrl = random.choice(temp_lst[min_idx:max_idx])
        if _gate in PGATES_2Q:
            if fix_params:
                gstr = _gate + "=" + str(_targ) + "=" + str(_ctrl) + "=%1.4f"%params_2q[i]
            else:
                gstr = _gate + "=" + str(_targ) + "=" + str(_ctrl) + "=param{}".format(num_params)
            num_params += 1
        else:
            gstr = _gate + "=" + str(_targ) + "=" + str(_ctrl) + "=nop"

        gate_strs.append(gstr)

    # mix 1q and 2q gates
    random.shuffle(gate_strs)
    q_string = ""
    for i in range(n_gates):
        q_string += gate_strs[i] + "@"
    return q_string[:-1], num_params


def _gen_random_idx(weights: list, rand_seed: int=None):
    '''
    Generate a random integer based on the weight.

    Args:
        :param weights: weights for the indices.
    Returns:
        A random index.
    Examples:
        >>> weight = [0.4, 0.6]
            Return 0 with probability 0.4 and 1 with probability 0.6.
    '''
    l_lst = len(weights)
    assert l_lst > 1

    _weights = numpy.array(weights)
    tot_weight = numpy.sum(_weights)
    assert tot_weight > 0

    _weights /= tot_weight
    # TODO maybe add random seed
    numpy.random.seed(rand_seed)
    rand_num = numpy.random.rand()
    left = 0.
    right  = _weights[0]
    for i in range(l_lst):
        if rand_num >= left and rand_num < right:
            rand_index = i
            break
        else:
            left += _weights[i]
            right += _weights[i+1]
    return rand_index

def _random_elem_from_lst(lst: list, qubit_targ: int=None,
                          max_dist: int=None, rand_seed: int=None):
    '''
    Pick one element from the list and then delete this element from the list.

    Args:
        :param lst: the list from which the element is selected and deleted.
    Kwargs:
        :param qubit_targ: the target qubit, only for 2-qubit gates
        :param max_dist: maximum distance of target and control qubits
    Returns:
        :int: the element int the list selected.
    '''
    # TODO: add random seed
    l_lst = len(lst)
    if qubit_targ is None or max_dist is None:
        random.seed(rand_seed)
        elem = random.choice(lst)
    elif max_dist >= l_lst:
        random.seed(rand_seed)
        elem = random.choice(lst)
    else:
        lst_ctrl = [el for el in lst if abs(el - qubit_targ) <= max_dist]
        random.seed(rand_seed)
        elem = random.choice(lst_ctrl)

    lst.remove(elem)
    return elem
