import numpy as np
import random
import logging
import pickle

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


class NeuralNetwork:

    def __init__(self, num_layers, num_neurons_per_layer_list, seed=400):
        logging.info("Creating a NN with {} layers having {} neurons respectively".format(num_layers,
                                                                                          num_neurons_per_layer_list))
        seed_everything(seed)
        self.seed = seed
        self.num_layers = num_layers
        self.num_neurons_per_layer_list = num_neurons_per_layer_list
        # The following self.weights and self.biases list contains layer-by-layer weights and biases as numpy arrays
        # For example, if we have an input layer with 784 neurons, two hidden layer with 8 neurons, and an output layer
        # with 10 neurons:
        # self.biases = [numpy array (8, 1), numpy array (8, 1), numpy array (10, 1)]
        # self.weights = [numpy array (784, 8), numpy array (8, 8), numpy array (8, 10)]
        self.biases = [np.random.randn(i, 1) for i in num_neurons_per_layer_list[1:]]
        self.weights = [np.random.randn(i, j)
                        for i, j in zip(num_neurons_per_layer_list[:-1], num_neurons_per_layer_list[1:])]
        pass

    @staticmethod
    def sigmoid(z):
        """
        :param z: numpy array (number of neurons, 1)
        :return: numpy array (number of neurons, 1)
        """
        sigmoid_z = np.zeros(z.shape)
        # TODO implement sigmoid
        sigmoid_z = 1 / (1 + np.exp(-z))
        # TODO implement sigmoid
        return sigmoid_z

    @staticmethod
    def cost_derivative(output_activations, y):
        """
        Derivative of cost w.r.t final activations
        :param output_activations: numpy array (number of neurons in output layer, 1)
        :param y: numpy array (number of neurons in output layer, 1)
        :return: numpy array (number of neurons in output layer, 1)
        """
        derivative_cost = np.zeros(output_activations.shape)
        # TODO calculate Derivative of cost w.r.t final activation
        derivative_cost = output_activations - y
        # TODO calculate Derivative of cost w.r.t final activation
        return derivative_cost

    def cost(self, training_data):
        """ Calculate cost (sum of squared differences) over training data for the current weights and biases
        :param training_data:
        list of 50000 samples, each item of the list is a tuple (x, y)
        x - input numpy array (784, 1)  y - labelled output numpy array (10, 1)
        :return: cost - float
        """
        logging.info("Calcuting costs...")
        cost = 0
        # TODO calculate costs
        for item in training_data:
            cost += 0.5*np.sum((self.forward_pass(item[0]) - item[1])**2)
        # TODO calculate costs
        logging.info("Calcuting costs complete...")
        return cost

    def sigmoid_derivative(self, z):
        """Derivative of the sigmoid function."""
        derivative_sigmoid = np.zeros(z.shape)
        # TODO calculate derivative of sigmoid function
        derivative_sigmoid = self.sigmoid(z)*(1 - self.sigmoid(z))
        # TODO calculate derivative of sigmoid function
        return derivative_sigmoid

    def forward_pass(self, x):
        """
        Perform forward pass and return the output of the NN.
        :param x: numpy array (784, 1)
        :return: numpy array (10, 1)
        """
        nn_output = np.zeros((self.num_neurons_per_layer_list[-1], 1))
        # TODO do a forward pass of the NN and return the final output
        
        z1 = np.matmul(self.weights[0].T,x) + self.biases[0]
        a1 = self.sigmoid(z1)
        z2 = np.matmul(self.weights[1].T, a1) + self.biases[1]
        a2 = self.sigmoid(z2)
        z3 = np.matmul(self.weights[2].T, a2) + self.biases[2]
        nn_output = self.sigmoid(z3)
        # TODO do a forward pass of the NN and return the final output
        return nn_output

    def mini_batch_GD(self, training_data, epochs, mini_batch_size, eta, test_data):
        """ Train the neural network using mini batch gradient descent
        :param training_data: list of tuples, where each tuple is a single labelled data point as follows.
         "(x - numpy array (784, 1), y (10, 1))"
        :param epochs: int
        :param mini_batch_size: int
        :param eta: learning rate float
        :param test_data: test data
        :return: cost_history - list (append cost after every epoch to this list, this will used to reward marks for
        gradient descent update step)
        """
        logging.info("Running mini_batch_GD")
        num_samples = len(training_data)
        costs_history = []
        for j in range(epochs):
            random.Random(self.seed).shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, num_samples, mini_batch_size)]
            for mini_batch in mini_batches:
                # TODO (1) Compute gradients for every data point in the mini batch using your implemented
                #  "back_propagation" function. Note that the above backward pass is for a single data point.
                #  We'll need to calculate this for all examples in a mini batch and then sum the gradients over this
                #  batch to do the gradient update step for all weights and biases.
                dw0 = 0
                dw1 = 0
                dw2 = 0
                db0 = 0
                db1 = 0
                db2 = 0
                for input in mini_batch:
                    dC_db, dC_dw = self.back_propagation(input[0],input[1])
                    dw0 += dC_dw[0]
                    dw1 += dC_dw[1]
                    dw2 += dC_dw[2]
                    db0 += dC_db[0]
                    db1 += dC_db[1]
                    db2 += dC_db[2]

                # TODO (2) Update biases and weights using the computed gradients
                self.weights[0] = self.weights[0] - (eta/mini_batch_size)*dw0
                self.weights[1] = self.weights[1] - (eta/mini_batch_size)*dw1
                self.weights[2] = self.weights[2] - (eta/mini_batch_size)*dw2
                self.biases[0] = self.biases[0] - (eta/mini_batch_size)*db0
                self.biases[1] = self.biases[1] - (eta/mini_batch_size)*db1
                self.biases[2] = self.biases[2] - (eta/mini_batch_size)*db2

            logging.info("After Epoch {}".format(j + 1))
            self.test_accuracy(test_data)
            cost_curr = self.cost(training_data)
            costs_history.append(cost_curr)
            logging.info("Cost: {}".format(cost_curr))
        return costs_history

    def back_propagation(self, x, y):
        """Compute gradients
        :param x: Input to the NN - numpy array (784, 1)
        :param y: Labelled output data - numpy array (10, 1)
        :return: tuple (dC_db, dC_dw) - list of numpy arrays, similar to self.biases and self.weights
        """
        # These list contain layer-by-layer gradients as numpy arrays, similar to self.biases and self.weights
        # For example, if we have an input layer with 784 neurons, two hidden layer with 8 neurons, and an output layer
        # with 10 neurons:
        # dC_db = [numpy array (8, 1), numpy array (8, 1), numpy array (10, 1)]
        # dC_dw = [numpy array (784, 8), numpy array (8, 8), numpy array (8, 10)]
        dC_db = [np.zeros(b.shape) for b in self.biases]
        dC_dw = [np.zeros(w.shape) for w in self.weights]

        # TODO (1) forward pass - calculate layer by layer z's and activations which will be used to calculate gradients
        z1 = np.matmul(self.weights[0].T,x) + self.biases[0]
        a1 = self.sigmoid(z1) #8x1
        z2 = np.matmul(self.weights[1].T, a1) + self.biases[1]
        a2 = self.sigmoid(z2) #8x1
        z3 = np.matmul(self.weights[2].T, a2) + self.biases[2]
        a3 = self.sigmoid(z3) #10x1

        # TODO (2) backward pass - calculate gradients starting with the output layer and then the hidden layers

        #for the output layer
        dC_dW3 = np.matmul(a2,(self.cost_derivative(a3,y)*self.sigmoid_derivative(z3)).T) #8x1 * 1*10
        dC_db3 = self.cost_derivative(a3,y)*self.sigmoid_derivative(z3)

        #for hidden layer 2
        # 10x1 and weights3 = weights[2] is 8x10 giving 8x1 da2
        dC_da2 = np.matmul(self.weights[2],self.cost_derivative(a3,y)*self.sigmoid_derivative(z3)) #np.sum(np.dot(self.cost_derivative(a3,y)*self.sigmoid_derivative(z3), self.weights[2].T),axis = 0).reshape(8,1) #10x8 matrix and take sum along the vertical axis to flatten into 1x8  :: 
        dC_dW2 = np.matmul(a1,(dC_da2 * self.sigmoid_derivative(z2)).reshape(1,-1)) #8x1 @ 1x8
        dC_db2 = dC_da2 * self.sigmoid_derivative(z2)   

        #for hidden layer 1
        dC_da1 = np.matmul(self.weights[1],dC_da2*self.sigmoid_derivative(z2))#np.sum(np.dot(self.cost_derivative(a2,y)*self.sigmoid_derivative(z2), self.weights[1].T),axis = 0).reshape(8,1) #784x8 matrix and take sum along the vertical axis to flatten into 1x8
        dC_dW1 = np.matmul(x,(dC_da1 * self.sigmoid_derivative(z1)).reshape(1,-1)) #784x1 @ 1x8
        dC_db1 = dC_da1 * self.sigmoid_derivative(z1)
        # TODO (3) Return the gradients in lists dC_db, dC_dw
        dC_db[0] = dC_db1
        dC_db[1] = dC_db2
        dC_db[2] = dC_db3

        dC_dw[0] = dC_dW1
        dC_dw[1] = dC_dW2
        dC_dw[2] = dC_dW3
        return dC_db, dC_dw

    def test_accuracy(self, test_data):
        acc_val = 0
        for x, y in test_data:
            if y == np.argmax(self.forward_pass(x)):
                acc_val += 1
        logging.info("Test accuracy {}".format(round(acc_val / len(test_data) * 100, 2)))


def load_data(file="data.pkl"):
    def vectorize(j):
        y = np.zeros((10, 1))
        y[j] = 1.0
        return y

    logging.info("loading data...")
    with open(file, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
        training_outputs = [vectorize(y) for y in training_data[1]]
        training_data = list(zip(training_inputs, training_outputs))
        test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
        test_data = list(zip(test_inputs, test_data[1]))
    logging.info("loaded data...")
    return training_data, test_data


def seed_everything(seed=0):
    np.random.seed(seed)


if __name__ == "__main__":
    data_training, data_test = load_data()
    net = NeuralNetwork(num_layers=4, num_neurons_per_layer_list=[784, 8, 8, 10])
    net.mini_batch_GD(data_training, 15, 50, 0.1, data_test)
