from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.result import Result
from typing import List

class ReUploadingPQC:

    def __init__(self, n_qubits: int, c_depth: int, backend, shots: int):
        # --- Circuit definition ---
        self.n_qubits = n_qubits
        self.c_depth = c_depth
        self.backend = backend
        self.shots = shots
        self.num_parameters = 0 # incremented when adding a parmeterized circuit.
                                # See the method create_parameterized_circuit.

        # prepare small circuits that are components to create the whole reuploading circuit
        self.parameterized_circuit_list: List[QuantumCircuit] = []
        self.parameters: List[Parameter] = []
        self.entangled_circuit_list: List[QuantumCircuit] = []
        self.input_circuit_list: List[QuantumCircuit] = []
        self.input_parameters: List[Parameter] = []
        self.prepare_small_circuits()

        # print(self.backend.name())
        # create the whole reuploading circuit
        self._circuit = self.compose_circuit()
        self._circuit = transpile(self._circuit)


    def compose_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for d in range(self.c_depth):
            qc.compose(self.parameterized_circuit_list[d], inplace=True)
            qc.compose(self.entangled_circuit_list[d], inplace=True)
            qc.barrier()
            qc.compose(self.input_circuit_list[d], inplace=True)
        qc.compose(self.parameterized_circuit_list[-1], inplace=True)
        if self.backend.name() == "qasm_simulator":
            qc.measure_all() # if you get a expectation value by sampling measurement results, you need to measure the qubits.
        return qc

    def prepare_small_circuits(self) -> None:
        for d in range(self.c_depth):
            self.append_created_circuit_to_list(type="parameter")
            self.append_created_circuit_to_list(type="entangle")
            self.append_created_circuit_to_list(type="input", depth=d)
        self.append_created_circuit_to_list(type="parameter")

    def append_created_circuit_to_list(self, type: str, depth: int = None) -> None:
        if type == "parameter":
            list = self.parameterized_circuit_list
            circuit = self.create_parameterized_circuit()
            params = self.parameters
        elif type == "entangle":
            list = self.entangled_circuit_list
            circuit = self.create_CZs_circuit()        
        elif type == "input":
            list = self.input_circuit_list
            circuit = self.create_input_circuit(depth)
            params = self.input_parameters
        else:
            print("error")

        # 生成した回路をリストで保存する
        list.append(circuit)
        # appendしたcircuitのparameterをクラス変数として保存する
        if type in ("parameter", "input"):
            for param in list[-1].parameters:
                params.append(param)

    def create_input_circuit(self, depth: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(Parameter(f"input_{depth}_{i}"), i) # input_depht_posofqubit
        return qc

    def create_parameterized_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.rx(Parameter("'" + str(self.num_parameters + qc.num_parameters) + "'"), i)
            qc.ry(Parameter("'" + str(self.num_parameters + qc.num_parameters) + "'"), i)
            qc.rz(Parameter("'" + str(self.num_parameters + qc.num_parameters) + "'"), i)
        self.num_parameters += qc.num_parameters # counting parameters
        return qc

    def create_CZs_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.cz(i, (i+1) % self.n_qubits)
        return qc

    def run(self,  x: List[float], thetas: List[float]) -> Result:
        params = self.parameters
        input_params = self.input_parameters
        keys = params + input_params
        values = thetas + x
        d = dict([(key, value) for key, value in zip(keys, values)])

        circuit_copy = self._circuit.copy()
        circuit_copy.assign_parameters(d, inplace = True)
        qobj = assemble(circuit_copy, backend=self.backend)
        job = self.backend.run(qobj, shots=self.shots)
        result = job.result()

        return result

    # it works only at jupyter notebook. Remove the "#" of the display method.
    def draw(self):
        qc = self._circuit
        fig = qc.draw(output='mpl')
        # display(fig)