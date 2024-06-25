import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_inputs = np.array([[0.5, 0.3, 0.3, 0.3], [0.4, 0.8, 0.9, 0.4], [0.4, 0.7, 0.5, 0.5],
                             [0.4, 0.9, 0.8, 0.6], [0.4, 0.65, 0.6, 0.7], [0.4, 0.4, 0.7, 0.7], [0.4, 0.8, 0.4, 0.9],
                             [0.4, 0.5, 0.8, 0.9], [0.4, 0.35, 0.4, 0.9], [0.4, 0.55, 0.8, 0.7]])

training_outputs = np.array([[0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.9, 0.9, 0.9, 0.7]]).T



np.random.seed(1)

synaptic_weights = 2 * np.random.random((4, 1)) - 1
print("Случайные веса: ")
print(synaptic_weights)

 #Метод обратного распространени
for i in range(20000):
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    if i == 1:
        print(outputs)
    err = training_outputs - outputs

    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))
    synaptic_weights += adjustments

print("Веса после обучения: ")
print(synaptic_weights)


print("Результат после обучения: ")
for out in outputs:
    print(round(float(out) * 10))

np.save('synaptic_weights.npy', synaptic_weights)

print("Synaptic weights saved to file.")



# Load the saved synaptic weights
synaptic_weights = np.load('synaptic_weights.npy')

print("Synaptic weights loaded from file.")

# Use the loaded model to make predictions
def predict(inputs):
    outputs = sigmoid(np.dot(inputs, synaptic_weights))
    return outputs

# Example usage
inputs = np.array([[0.4, 0.65, 0.5, 0.5]])
output = predict(inputs)
print("Output:", output)


