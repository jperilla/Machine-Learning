import math
import random

from machineLearningUtilities import get_num_similarities


class BackpropagationNeuralNetwork:
    """
    This class performs the backpropagation algorithm to train an artificial neural network
    """
    def __init__(self, features, x_train, x_train_classes, x_test, x_test_classes, num_hidden=1):
        self.features = features
        self.training_classes = x_train_classes
        self.train = x_train
        self.test = x_test
        self.test_classes = x_test_classes
        self.num_iterations = 20
        self.eta = 0.01  # learning rate
        self.network = list()
        self.num_hidden_layers = num_hidden

    def learn(self):
        """
        This function runs the backpropagation algorithm on the training set provided
        """
        print("Running Backpropagation...")

        print("Initializing Network...")
        num_inputs = len(self.train.columns) - 1
        hidden = [{'w': [random.uniform(-1.0, 1.0) for i in range(num_inputs + 1)]} for i in range(self.num_hidden_layers)]
        self.network.append(hidden)
        num_outputs = len(set([c for c in self.training_classes]))
        output = [{'w': [random.uniform(-1.0, 1.0) for i in range(self.num_hidden_layers + 1)]} for i in range(num_outputs)]
        self.network.append(output)

        print("Training Network...")
        for iteration in range(self.num_iterations):
            error = 0
            for x_index, row in self.train.iterrows():
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(num_outputs)]
                expected[int(self.training_classes[x_index])] = 1
                error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row)
            print('>iteration=%d, learning rate=%.3f, error=%.3f' % (iteration, self.eta, error))

    def make_predictions(self):
        predictions = list()
        for i, row in self.test.iterrows():
            prediction = self.predict(row)
            predictions.append(prediction)

        print("Predicted Classes = ")
        print(predictions)

        print("Expected Classes = ")
        print(list(self.test_classes))

        return get_num_similarities(predictions, self.test_classes) / len(self.test_classes) * 100

    def predict(self, row):
        """
        This function determines accuracy of model using the test data set
        and applying the linear function using the weights
        """
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

    @staticmethod
    def sigmoid(x):
        """
        This function calculates the logistic sigmoid of x and is used as the transfer function
        :param x: The value
        :return: The logistic sigmoid
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return float('inf')

    @staticmethod
    def activate(weights, inputs):
        """
        This function performs activation by calculating the weighted sum of the
        inputs plus the bias
        :param weights: the current weight vector with bias as the last element
        :param inputs: a vector of inputs
        :return: The activation
        """
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * float(inputs[i])
        return activation

    @staticmethod
    def transfer_derivative(y):
        """
        This function calculates the derivative to be used as the slope of an output value
        :param y: The output value
        :return: The derivative / slope
        """
        return y * (1.0 - y)

    def forward_propagate(self, inputs):
        """
        This function implements forward propagation, the inputs are taken and outpuss
        are calculated by first running the activation function and then the transfer (sigmoid)
        function using the activation, each new input is appended to the output layer and
        returned
        :param inputs: The input vector
        :return: The output layer
        """
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                x = self.activate(neuron['w'], inputs)
                neuron['output'] = self.sigmoid(x)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        """
        This function performs backpropagation by calcuating error betweeen the expected outputs
        and the actual outputs of forward propagation. These error calcuations are propagated back
        throught the network from output to hidden layer, looking for where the error occurred.
        :param expected:
        :return:
        """
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['w'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * self.transfer_derivative(neuron['output'])

    def update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['w'][j] += self.eta * float(neuron['delta']) * float(inputs[j])
                neuron['w'][-1] += self.eta * float(neuron['delta'])


