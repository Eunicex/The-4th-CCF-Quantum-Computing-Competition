from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.tensor import tensor
from pyvqnet.nn.module import Module
from pyvqnet.nn import Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CrossEntropyLoss
from pyqpanda import *
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from data_preprocess import data_preprocess
from pyvqnet.qnn.quantumlayer import QuantumLayer
import pyqpanda as pq

n_qubits = 6
q_depth  = 3

def load_data(data_csv):
    """
    加载并转换数据为 QTensor
    """
    features, labels = data_preprocess(data_csv)
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return QTensor(X), QTensor(y)

def Q_H_layer(qubits, nqubits):
    """Layer of single-qubit Hadamard gates.
    """
    circuit = pq.QCircuit()
    for idx in range(nqubits):
        circuit.insert(pq.H(qubits[idx]))
    return circuit

def Q_RY_layer(qubits, w):
    """
    Layer of parametrized qubit rotations around the y axis.
    """
    circuit = pq.QCircuit()
    for idx, element in enumerate(w):
        circuit.insert(pq.RY(qubits[idx], element))
    return circuit

def Q_entangling_layer(qubits, nqubits):
    """
    Layer of CNOTs followed by another shifted layer of CNOT.
    """
    circuit = pq.QCircuit()
    for i in range(0, nqubits - 1,
                    2):  # Loop over even indices: i=0,2,...N-2
        circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
    for i in range(1, nqubits - 1,
                    2):  # Loop over odd indices:  i=1,3,...N-3
        circuit.insert(pq.CNOT(qubits[i], qubits[i + 1]))
    return circuit

def quantum_net(q_inputs, q_weights_flat, *_):
    """单样本电路, 返回长度4 的 Z 期望值"""
    machine = pq.CPUQVM(); 
    machine.init_qvm()
    qs = machine.qAlloc_many(n_qubits)

    circ = pq.QCircuit()

    circ.insert(Q_H_layer(qs, n_qubits)) 
    circ.insert(Q_RY_layer(qs, q_inputs))

    q_weights = q_weights_flat.reshape(q_depth, n_qubits)
    for layer in range(q_depth):
        # entangle
        circ.insert(Q_entangling_layer(qs, n_qubits))
        circ.insert(Q_RY_layer(qs, q_weights[layer]))

    prog = pq.QProg(); 
    prog.insert(circ)

    out = []
    for i in range(n_qubits):
        op_str = "Z" + str(i)
        op_map = pq.PauliOperator(op_str, 1)
        hamiltion = op_map.toHamiltonian(True)
        exp = machine.get_expectation(prog, hamiltion, qs)
        out.append(exp)

    return out

class AirQualityQuantumNN(Module):
    def __init__(self):
        super().__init__()
        self.pre  = Linear(9, n_qubits)
        self.post = Linear(n_qubits, 4)
        self.vqc  = QuantumLayer(quantum_net, q_depth*n_qubits, "cpu", n_qubits, n_qubits)

    def forward(self, x):
        x = tensor.tanh(self.pre(x)) * np.pi/2
        x = self.vqc(x)
        return self.post(x)
    
def quantum_model_train():
    X_train, y_train = load_data('./train_data.csv')

    num_epoch = 50
    # batch = 128
    lr = 0.1

    model = AirQualityQuantumNN()
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for epoch in range(num_epoch):
        if epoch % 10 == 0 and epoch != 0:
            optimizer.lr *= 0.5

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(y_train, outputs)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}")

    return model

def model_test(model):
    X_test, y_test = load_data('./test_data.csv')

    outputs = model(X_test)
    predicted = np.argmax(outputs.data, axis=1)
    y_true = y_test.to_numpy()

    accuracy = accuracy_score(y_true, predicted)
    f1 = f1_score(y_true, predicted, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    with open('quantum_model_results.txt', 'w') as f:
        f.write(f"{accuracy:.4f} {f1:.4f}")

if __name__ == "__main__":
    model = quantum_model_train()

    model_test(model)