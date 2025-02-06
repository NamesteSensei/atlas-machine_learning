import numpy as np
class DeepNeuralNetwork:
    
    def __init__(self, nx, layers):
        
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if (not isinstance(layers, list) or len(layers) == 0:
            raise TypedError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev_layer_size = nx  # Tracks size of the previous layer
        for layer in range(1, self.__L + 1):
            self.__weights[f"W{layer}"] = (
                np.random.randn(layers[layer - 1], prev_layer_size) *
                np.sqrt(2 / prev_layer_size)
            )
            self.__weights[f"b{layer}"] = np.zeros((layers[layer - 1], 1))
            prev_layer_size = layers[layer - 1]  # Update next layer
    @property
    def L(self):
        
        return self.__L
    @property
    def cache(self):
        
        return self.__cache
    @property
    def weights(self):
        
        return self.__weights
    def forward_prop(self, X):
        
        self.__cache["A0"] = X  # Store input data
        prev_activation = X
        for layer in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f"W{layer}"], prev_activation) + self.__weights[f"b{layer}"]
            prev_activation = 1 / (1 + np.exp(-Z))  # Sigmoid activation
            self.__cache[f"A{layer}"] = prev_activation
        return prev_activation, self.__cache
