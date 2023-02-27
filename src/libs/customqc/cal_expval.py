import numpy as np


from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import Statevector
from qiskit.result import Result

def cal_expectation_values(result: Result) -> np.ndarray:
    """
    return numpy array([float. float])
    """

    if result.backend_name == "qasm_simulator":
        expectation_values = cal_expectation_values_by_qasm_simulator(result)
    elif result.backend_name == "statevector_simulator":
        expectation_values = cal_expectation_values_by_statevector_simulator(result)

    return expectation_values

def cal_expectation_values_by_qasm_simulator(result: Result) -> np.ndarray:
    """
    return numpy array([float. float])
    """

    result_dic = result.get_counts()

    counts = np.array(list(result_dic.values()))
    states = np.array(list(result_dic.keys()))

    counts_for_z0z1 = 0
    counts_for_z2z3 = 0
    shots = 0
    for state, count in zip(states, counts):
        shots += count

        sign_z0z1 =int(state[0]) + int(state[1])
        if (sign_z0z1%2 == 0):
            counts_for_z0z1 += count
        else:
            counts_for_z0z1 -= count

        sign_z2z3 =int(state[2]) + int(state[3])
        if (sign_z2z3%2 == 0):
            counts_for_z2z3 += count
        else:
            counts_for_z2z3 -= count

    expectation_values = np.array([float(counts_for_z0z1), float(counts_for_z2z3)])/shots

    return expectation_values

def cal_expectation_values_by_statevector_simulator(result: Result) -> np.ndarray:
    """
    return numpy array([float. float])
    """

    result_statevec = Statevector(result.get_statevector())
    obs01 = Pauli('ZZII')
    obs02 = Pauli('IIZZ')
    expectation01 = result_statevec.expectation_value(obs01)
    expectation02 = result_statevec.expectation_value(obs02)

    expectation_values = np.array([np.real(expectation01), np.real(expectation02)])

    return expectation_values