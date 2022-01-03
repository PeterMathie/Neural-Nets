from typing import ByteString
import numpy as np


#1. variance of Wavelet Transformed image (continuous) 
#2. skewness of Wavelet Transformed image (continuous) 
#3. curtosis of Wavelet Transformed image (continuous) 
#4. entropy of image (continuous) 
#5. class (integer) 

#opens the dataset file 
#adds values, seperated by newlines and commas to an array X of floats
#init length of F many weights
def get_inputs():
    f = open("data_banknote_authentication.txt", "r")
    X = [[float(num) for num in line.split(',')] for line in f if line.strip() != "" ]
    return X
    
#makes an length x array of random numbers within a range
def init_weights(x):
    W = np.random.uniform(low=-1, high=1, size=(x,))
    return W

#returns a random number between -1 and 1
def init_bias():
    b = np.random.uniform(low=-1, high=1)
    return b

#makes an length x array of random numbers within a range
def random_int(lower_bound, upper_bound):
    r = np.randint(low=lower_bound, high=upper_bound, dtype=int)
    return r
def random_float(lower_bound, upper_bound):
    r = np.random.uniform(low=lower_bound, high=upper_bound)
    return r
def random_int_arr(lower_bound, upper_bound, dimX, dimY):
    if(dimY == 0):
        r = np.randint(low=lower_bound, high=upper_bound, size=(dimX,))
    else:
        r = np.randint(low=lower_bound, high=upper_bound, size=(dimX,dimY))

    return r
def random_float_arr(lower_bound, upper_bound, dimX, dimY):
    if(dimY == 0):
        r = np.random.uniform(low=lower_bound, high=upper_bound, size=(dimX,))
    else:
        r = np.random.uniform(low=lower_bound, high=upper_bound, size=(dimX,dimY))
    return r


#Activation_Functions class holds all relevant functions for the implemented activation functions
#neurons are made with an instance of Activation_Functions as a parameter
class Activation_Functions:

    #initialise a neuron's func but passing a string. The string is the name of the function
    def __init__(self, function):

        self.function = function

    #nerons will call the activate function through their funciton parameter
    #This function acts as a hub where you can call the relevant activation function 
    def activate(self, linear_vector):
        if(np.floor(self.function)==0):
            return self.sigmoid(linear_vector)
        if(np.floor(self.function)==1):
            return self.ReLU(linear_vector)
        if(np.floor(self.function)==2):
            return self.tanh(linear_vector)
        if(np.floor(self.function)==3):
            return self.softPlus(linear_vector)

    #normalises the value of out linear model between 1 and 0 using the sigmoid function
    def sigmoid(self,linear_vector):
        activation_vector = []
        for z in linear_vector:
            val = 1/(1+np.power(2.71828,-z))
            activation_vector.append(val)
        return activation_vector

    #normalises the value of out linear model between 1 and 0 using the ReLU function
    #for values<0, set value = 0
    #for values>0, set value = value
    def ReLU(self, linear_vector):
        activation_vector = []
        for z in linear_vector:
            if(z<0):
                activation_vector.append(0)
            else:
                activation_vector.append(z)
        return activation_vector
    
    #normalises the value of out linear model between 1 and -1 using the tanh function
    def tanh(self, linear_vector ):
        activation_vector = []
        for z in linear_vector:
                activation_vector.append(np.tanh(z))
        return activation_vector    

    #normalises the value of out linear model between 1 and 0 using the softPlus function
    def softPlus(self, linear_vector):
        activation_vector = []
        for z in linear_vector:
            activation_vector.append(np.log(1+np.power(2.71828,z)))
        return activation_vector

    #normalises the value of out linear model between 1 and 0 using the binary step function
    def binary_step(self, linear_vector):
        activation_vector = []
        for z in linear_vector:
            if(z<0):
                activation_vector.append(0)
            else:
                activation_vector.append(1)
#Neuron class defines the neuron's hyperparameters and the linear model
class Neuron:
    def __init__(self, weights_vector, bias, activation_func, layer):
        # a vector containing all the weights of the neuron
        self.weights_vector = np.array(weights_vector)
        # the neuron's bias
        self.bias = bias
        # string that tells us what activation function we are using
        self.activation_func = Activation_Functions(activation_func)
        # the layer yo ucan find the neuron on
        self.layer = layer
        # init as empty but will store the vector produced from the linear model
        self.linear_vector = []
        # init as empty but will store the vector produced from the activation function
        self.output = []

    #transpose the weight vector for matrix multiplication
    #multiplies the transposed weights by input data
    #adds bias to all values in Z
    # Z = weights_tansposed * input_vector
    def linear(self, input_vector):
        vector = []
        #print("\n-------INPUT VEECTOR-------\n",np.array(input_vector))
        #print("\n-------WEIGHTS VEECTOR-------\n",np.array(self.weights_vector))
        
        weights_tansposed = np.transpose(self.weights_vector)
        # first layer of the net reads data from the file which has the class attribute
        # to ignore the class attribute we read the first 4 elements from each row
        # method used requires a numpy array
        if(self.layer == 0):
            linear_vector = np.matmul(input_vector[::,0:4],weights_tansposed)
        else:
            linear_vector = np.matmul(input_vector,weights_tansposed)
        # add the bias to all in the linear vector
        for l in range(len(linear_vector)):
            vector.append(linear_vector[l]+self.bias)
        # update the value of the linear vector in the neuron
        self.linear_vector = vector
        # probably bad for storage optimisation to store the weight vector again, but its old code from before the datastructure for a neuron was complete so could break something if i removed it (dont think it would though)
        return vector
    
# defines the structure of the net and relevant classes 
class Net:
    def __init__(self, num_inputs, num_hidden_layers, nodes_in, nodes_hidden, nodes_out, functions):
        
        #ftiness of this net, measured using the cost fucntion (lower is better)
        self.fitness = 999
        #number of inputs to the input layer
        self.num_inputs = num_inputs
        #number of nodes in the input layer
        self.nodes_in = nodes_in
        #number of hidden layers
        self.num_hidden_layers = num_hidden_layers
        #number of nodes in each hidden layer, stored as an array so the net can have any structure with a input, hidden and output layer
        self.nodes_hidden = nodes_hidden
        #number of nodes in the output layer
        self.nodes_out = nodes_out
        
        # The activation functions for each layer
        # first layer activation function
        self.first_function = functions[0]
        # array of the hidden layers functions
        self.hidden_functions = functions[1]
        # last layer activation function
        self.out_function = functions[2]

        # array's holding the output of each layer, used in the updating of the wieghts where we used the previous layer output
        self.first_layer_outputs = []
        self.hidden_layers_outputs = []
        self.final_layer_outputs = []

        # with all parameters set you can initialise the net
        self.net = self.init_structure()
        # for quality of life, can access a layer with net.input_layer, ect
        self.input_layer = self.net[0]
        self.hidden_layers = self.net[1]
        self.output_layer = self.net[2]

    # functionality for the creation of hidden layers
    # creates a 2D array where each ith row is the length the value in the ith postion nodes_hidden array
    def init_hidden_structure(self,nodes_hidden):
        # hidden_layers holds all the rows 
        hidden_layers = []
        for i in nodes_hidden:
            # define a 1D numpy array of type Neuron and length i
            row = np.empty(dtype=Neuron,shape=i)
            hidden_layers.append(row)
        #return empty array of correct dimensions
        return hidden_layers
        
    def init_structure(self):
        # define sizes of each row as arrays that will store neurons 
        input_layer = np.empty(dtype=Neuron, shape=(self.nodes_in))
        hidden_layers = self.init_hidden_structure(self.nodes_hidden) 
        output_layer = np.empty(dtype=Neuron, shape=(self.nodes_out))

        # for make nodes_in many neurons and add them to the input_layer array
        for i in range(self.nodes_in):
            weights_vector = np.array(init_weights(self.num_inputs))
            bias = init_bias()
            input_layer[i] = Neuron(weights_vector, bias, self.first_function, 0)

        # make as many neurons as the value in each index of the hidden_layers array
        for j in range(0,self.num_hidden_layers):
            for k in range(self.nodes_hidden[j]):
                if(j == 0):
                    #generate a weight for each input from the previous layer
                    #first case: as many inputs as there are input nodes
                    weights_vector = np.array(init_weights(self.nodes_in))
                else:
                    #generate a weight for each input from the previous layer
                    #every other case: as many inputs as there are hidden nodes in the previous layer
                    weights_vector = np.array(init_weights(self.nodes_hidden[j-1]))
                    bias = init_bias()
                hidden_layers[j][k] = Neuron(weights_vector, bias, self.hidden_functions[j], j+1)

        # make the node_out many nodes for the output layer
        for l in range(self.nodes_out):
            #as many wieghts as nodes in one hidden layer
            weights_vector = np.array(init_weights(self.nodes_hidden[j]))
            bias = init_bias()
            output_layer[l] = Neuron(weights_vector, bias, self.out_function,(j+2))
        
        # add layers of neurons to the datastructure, to be acessed later
        net = [input_layer, hidden_layers, output_layer]
        return net
    # Apply the cross entropy loss function to predicted values vector (activation_vector) and actual values vector (input_vector[x][4])
    # Gives us a number describing how far off out prediction is from the actual value
    def loss(self,activation_vector,input_vector):
        loss_vector = []
        for x in range(0, len(activation_vector)):
            real = input_vector[x][4]
            predicted = activation_vector[x] 
            # the +0.00001 prevents any divide by 0 errors
            loss_vector.append(-(real*np.log(predicted+0.00001)+(1-real)*np.log(1-predicted+0.00001)))
        return loss_vector
    
    # cost function, gives us a single number to describe the performance of our neural network. 
    # When this number stops changing we have converged and can stop triaing the net
    def cost(self, loss_vector):
        return np.mean(loss_vector)

    #returns the proportion of values in the activation vector that are less than 0.2 from the actual value
    def error(self, activation_vector, input_vector):
        wrong_prediction = 0
        for i in range(len(activation_vector)):
            dx = np.absolute(input_vector[i][4] - activation_vector[i])
            if(dx > 0.2):
                wrong_prediction = wrong_prediction + 1
        error = wrong_prediction/len(activation_vector)
        return error

    # forward pass stage of the ANN
    def forward_pass_first(self, input_vector):
        activation_vector = []
        # for each node in the first layer
        for node in self.input_layer:
            #the linear regression to get a vector to be passed through the sigmoid function
            node.linear_vector = node.linear(input_vector)
            #compresses the linear model to values between 0 and 1 based on the sigmoid function
            node.output = node.activation_func.activate(node.linear_vector)
            #append activation vector with the node's output
            activation_vector.append(node.output)
        #rotate 90 degrees so the linear function can read it 
        activation_vector = np.transpose(activation_vector)
        self.first_layer_outputs = activation_vector
        #print("\n-------LINEAR VECTOR INPUT LAYER-------\n",np.array(node.linear_vector))
        #print("\n-------WEIGHTS VEECTOR INPUT-------\n",np.array(node.weights_vector))
        return activation_vector

    #similar to first layer's forward propagation but loops through mutiple hidden layer
    def forward_pass_hidden(self, activation_vector): 
        layers_vector = []
        for layer in self.hidden_layers:
            hidden_vector = []  
            for node in layer:
                #the linear regression to get a vector to be passed through the sigmoid function
                node.linear_vector = node.linear(activation_vector)
                #compresses the linear model to values between 0 and 1 based on the sigmoid function
                node.output = node.activation_func.activate(node.linear_vector)
                hidden_vector.append(node.output)
            activation_vector = np.transpose(hidden_vector)
            layers_vector.append(activation_vector)
        self.hidden_layers_outputs = layers_vector
        #print("\n-------LINEAR VECTOR HIDDEN-------\n",np.array(node.linear_vector))
        #print("\n-------WEIGHTS VEECTOR INPUT-------\n",np.array(node.weights_vector))
        return activation_vector
    
    #similar to the first layer's activation vector but takes the output of the last hidden layer as an input
    def forward_pass_final(self, activation_vector):
        layer_vector = []
        for node in self.output_layer:
            #the linear regression to get a vector to be passed through the sigmoid function
            node.linear_vector = node.linear(activation_vector)
            #compresses the linear model to values between 0 and 1 based on the sigmoid function
            node.output = node.activation_func.activate(node.linear_vector)
            layer_vector.append(node.output)
        self.final_layer_outputs = layer_vector
        #print("\n-------LINEAR VECTOR FINAL-------\n",np.array(node.linear_vector))
        #print("\n-------WEIGHTS VEECTOR INPUT-------\n",np.array(node.weights_vector))
        return node.output
    
    #
    def fitness_func(self, loss_vector):
        cost = self.cost(loss_vector)
        #print(cost, 1/(cost*cost))
        return 1/(cost*cost)

    def run(self,input_vector):
        #iterate through the layers and populate values for the activation functions
        #output from one function is the input to the next
        activation_vector = self.forward_pass_first(input_vector)
        activation_vector = self.forward_pass_hidden(activation_vector)
        activation_vector = self.forward_pass_final(activation_vector)

        #find the cross entropy loss
        loss_vector = self.loss(activation_vector,input_vector)

        #measure the new cost in j, these will be used to measure the fitness of this net
        
        self.fitness = self.cost(loss_vector)
        #the error for printing
        e = round(self.error(activation_vector,input_vector) * 100, 2)

class Particle:
    def __init__(self, num_inputs, net, alpha, beta, gamma, delta, epsilon):

        self.informants = []

        #init optimal value as a large number
        self.optima = 999

        #upper limits for obtaining stochastic congative bias, social bias and jump size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        # define particle to be the structure of the network
        # this holds all of the data we need, so no need to make another vector for its postion in the
        # there is also irrelevant data in here so not great for space complexity 
        self.net = net
        self.input_layer = net.input_layer
        self.hidden_layers = net.hidden_layers
        self.output_layer = net.output_layer

        #init best net we've seen so far as original net
        self.best_net = net
        self.best_informant_index = 0
        lower_limit =-1
        upper_limit = 1
        # vector will hold all the velocities for each hyperparameter we're changing in the input layer
        #  keeping the input layer seperate because it was taught that way in the lectures and I'm not going back and
        #  reformatting all this code for slight performance inprovements
        self.input_velocities_weights  = random_float_arr(lower_limit, upper_limit, len(self.input_layer), num_inputs)
        self.input_velocities_biases   = random_float_arr(lower_limit, upper_limit, len(self.input_layer), 0)
        self.input_velocities_function = random_float(-0.1,0.1)

        # vector will hold all the velocities for each hyperparameter we're changing in the hidden layers
        self.hidden_velocities_weights = []
        self.hidden_velocities_biases = []
        self.hidden_velocities_function = []

        for i in range(len(self.hidden_layers)):
            if(i==0):
                # first hidden layer takes its input from the input layer
                self.hidden_velocities_weights.append(random_float_arr(lower_limit, upper_limit, len(self.hidden_layers[i]), len(self.input_layer)))
            else:
                # every layer after the first hidden takes from the previous hidden layer
                self.hidden_velocities_weights.append(random_float_arr(lower_limit, upper_limit, len(self.hidden_layers[i]), len(self.hidden_layers[i-1])))
            self.hidden_velocities_biases.append(random_float_arr(lower_limit, upper_limit, len(self.hidden_layers[i]), 0))
            self.hidden_velocities_function.append(random_float(-0.1,0.1))

        self.output_velocities_weights = random_float_arr(lower_limit, upper_limit, len(self.hidden_layers[-1]), 0)
        self.output_velocities_biases = random_float_arr(lower_limit, upper_limit, len(self.output_layer), 0)
        self.output_velocities_function = random_float(-0.1,0.1)

    #adds particle to informant list
    def set_informants(self,informant):
        self.informants.append(informant)


    def access_fitness(self, input_vector):
        self.net.run(input_vector) 
        #if the current net has a better fitness than the previous best_net, update best_net to be the current net
        if(self.net.fitness < self.optima):
            self.best_net = self.net
            self.optima = self.net.fitness
        self.get_local_optima()

    #returns the index of the optimal informant
    def get_local_optima(self):
        for i in range(0, len(self.informants)):
            if(self.informants[i].optima < self.informants[self.best_informant_index].optima):
                self.best_informant_index = i

    def update_velocites(self, global_optima):

        index = self.best_informant_index
        cognitive = random_float(0,self.beta )
        social    = random_float(0,self.gamma)
        global_   = random_float(0,self.delta)

        #update the activation function
        self.input_velocities_function = self.alpha * self.input_velocities_function
        + social    * ( self.informants[index].best_net.first_function - self.net.first_function )
        + global_   * ( global_optima.best_net.first_function          - self.net.first_function )
        + cognitive * ( self.best_net.first_function                   - self.net.first_function ) 

        # update weights for the first layer
        for i in range(len(self.input_layer)): #for each node in the input layer

            #updates velocities for bias for input layer
            self.input_velocities_biases[i] = self.alpha * self.input_velocities_biases[i]
            + social    * ( self.informants[index].best_net.input_layer[i].bias - self.net.input_layer[i].bias )
            + global_   * ( global_optima.best_net.input_layer[i].bias          - self.net.input_layer[i].bias )
            + cognitive * ( self.best_net.input_layer[i].bias                   - self.net.input_layer[i].bias ) 

            #updates velocities for weights for input layer
            for j in range(len(self.input_velocities_weights[i])-1):
                self.input_velocities_weights[i][j] = self.alpha * self.input_velocities_weights[i][j]
                + social    * ( self.informants[index].best_net.input_layer[i].weights_vector[j] - self.net.input_layer[i].weights_vector[j] )
                + global_   * ( global_optima.best_net.input_layer[i].weights_vector[j]          - self.net.input_layer[i].weights_vector[j] )
                + cognitive * ( self.best_net.input_layer[i].weights_vector[j]                   - self.net.input_layer[i].weights_vector[j] ) 

        # update weights for the hidden layers
        for x in range(len(self.hidden_layers)):                                # through layers
            
            #updates velocities for hidden layer functions
            self.hidden_velocities_function[x] = self.alpha*self.hidden_velocities_function[x]
            + social    * ( self.informants[index].best_net.hidden_functions[x] - self.net.hidden_functions[x] )
            + global_   * ( global_optima.best_net.hidden_functions[x]          - self.net.hidden_functions[x] )
            + cognitive * ( self.best_net.hidden_functions[x]                   - self.net.hidden_functions[x] ) 

            #updates velocities for bias
            for y in range(len(self.hidden_layers[x])):                         # through nodes in layer    
                self.hidden_velocities_biases[x][y] = self.alpha * self.hidden_velocities_biases[x][y]
                + social    * ( self.informants[index].best_net.hidden_layers[x][i].bias - self.net.hidden_layers[x][y].bias )
                + global_   * ( global_optima.best_net.hidden_layers[x][y].bias          - self.net.hidden_layers[x][y].bias )
                + cognitive * ( self.best_net.hidden_layers[x][y].bias                   - self.net.hidden_layers[x][y].bias ) 

                #updates velocities for weights
                for z in range(len(self.hidden_velocities_weights[x][y])):      # through weights in nodes
                    self.hidden_velocities_weights[x][y][z] = self.alpha * self.hidden_velocities_weights[x][y][z]
                    + social    * ( self.informants[index].best_net.hidden_layers[x][y].weights_vector[z] - self.net.hidden_layers[x][y].weights_vector[z] )
                    + global_   * ( global_optima.best_net.hidden_layers[x][y].weights_vector[z]          - self.net.hidden_layers[x][y].weights_vector[z] )
                    + cognitive * ( self.best_net.hidden_layers[x][y].weights_vector[z]                   - self.net.hidden_layers[x][y].weights_vector[z] ) 

        #updates velocities for functions on final layer
        self.output_velocities_function = self.alpha * self.output_velocities_function
        + social    * ( self.informants[index].best_net.out_function - self.net.out_function ) 
        + global_   * ( global_optima.best_net.out_function          - self.net.out_function )
        + cognitive * ( self.best_net.out_function                   - self.net.out_function ) 
        
        for a in range(len(self.output_layer)):
            #updates velocities for bias for final layer
            self.output_velocities_biases[a] = self.alpha * self.output_velocities_biases[a]
            + social    * ( self.informants[index].best_net.output_layer[a].bias - self.net.output_layer[a].bias )
            + global_   * ( global_optima.best_net.output_layer[a].bias          - self.net.output_layer[a].bias )
            + cognitive * ( self.best_net.output_layer[a].bias                   - self.net.output_layer[a].bias ) 
            
            #updates velocities for weights for final layer
            for b in range(len(self.output_velocities_weights)):
                self.output_velocities_weights[b] = self.alpha * self.output_velocities_weights[b] 
                + social    * ( self.informants[index].best_net.output_layer[a].weights_vector[b] - self.net.output_layer[a].weights_vector[b] )
                + global_   * ( global_optima.best_net.output_layer[a].weights_vector[b]          - self.net.output_layer[a].weights_vector[b] )
                + cognitive * ( self.best_net.output_layer[a].weights_vector[b]                   - self.net.output_layer[a].weights_vector[b] ) 

    def update_values(self):

        # update values for the first layer
        self.net.first_function = self.epsilon * self.input_velocities_function + self.net.first_function
        for i in range(len(self.input_layer)): #for each node in the input layer
            self.net.input_layer[i].bias = self.epsilon * self.input_velocities_biases[i] + self.net.input_layer[i].bias
            for j in range(len(self.input_velocities_weights[i])-1):
                self.net.input_layer[i].weights_vector[j] = self.epsilon * self.input_velocities_weights[i][j] + self.net.input_layer[i].weights_vector[j]

        # update weights for the hidden layers
        for x in range(len(self.hidden_layers)):                                # through layers
            self.net.hidden_functions[x] = self.epsilon * self.hidden_velocities_function[x] + self.net.hidden_functions[x]
            for y in range(len(self.hidden_layers[x])):                         # through nodes in layer
                self.net.hidden_layers[x][y].bias = self.epsilon * self.hidden_velocities_biases[x][y] + self.net.hidden_layers[x][y].bias
                for z in range(len(self.hidden_velocities_weights[x][y])):      # through weights in nodes
                    self.net.hidden_layers[x][y].weights_vector[z] = self.epsilon * self.hidden_velocities_weights[x][y][z] + self.net.hidden_layers[x][y].weights_vector[z]

        self.net.out_function = self.epsilon * self.output_velocities_function + self.net.out_function
        # update weights for the final layer
        for a in range(len(self.output_layer)):
            # updates the bias
            self.net.output_layer[a].bias = self.epsilon * self.output_velocities_biases[a] + self.net.output_layer[a].bias
            # update the weights
            for b in range(len(self.output_velocities_weights)):
                self.net.output_layer[a].weights_vector[b] = self.epsilon * self.output_velocities_weights[b] + self.net.output_layer[a].weights_vector[b]


def init_particles(swarm_size, alpha, beta, gamma, delta, epsilon, num_inputs, num_hidden_layers, nodes_in, nodes_hidden, nodes_out, functions):
    #init the required number of particles and their nets
    nets = np.empty(swarm_size, dtype = Net)
    particles = np.empty(swarm_size, dtype = Particle)
    for i in range(swarm_size):
        nets[i] = Net(num_inputs, num_hidden_layers, nodes_in, nodes_hidden, nodes_out, functions)
        particles[i] = Particle(num_inputs, nets[i], alpha, beta, gamma, delta, epsilon )

    #randomly select ~1/4 of the particles for each particles' informant
    num_informants = np.ceil(swarm_size/20)
    for i in range(swarm_size):
        informants = np.random.choice(swarm_size, int(num_informants), replace = False)
        for inforamnt in informants:
            particles[i].set_informants(particles[inforamnt])

    return particles

def main( ):
    #HYPERPARAMETERS 
    #the input vector from our dataset
    input_vector = np.array(get_inputs())
    #the number of attributes in out dataset
    num_inputs = len(input_vector[0])-1

    #the nodes for the first layre
    nodes_in = 4
    #the nodes for the hidden layers, each layer can have any number of nodes
    nodes_hidden = [4]
    #nodes for the output layer (has to be 1)
    nodes_out = 1

    #number of hidden layers we want
    num_hidden_layers = len(nodes_hidden)

    #activation functions for thier respective layers
    # "sigmoid" = 0
    # "ReLU" = 1
    # "tanh" = 2
    # "softPlus" = 3
    first_function = 0
    hidden_functions = [1,0,1]
    out_function = 0
    functions = [first_function,hidden_functions,out_function]

    #proportion of velocity to be retained
    alpha = 0.9
    #proportion of personal best to be retained
    beta = 1.3
    #proportion of informants to be retained
    gamma = 1.2
    #proportion of global best to be retained
    delta = 1.5
    #jump size of particle
    epsilon = 1.1

    swarm_size = 100

    particles = init_particles(swarm_size, alpha, beta, gamma, delta, epsilon, num_inputs, num_hidden_layers, nodes_in, nodes_hidden, nodes_out, functions)
    
    global_optima = particles[0]
    
    time = 5
    # runs the swarm for time epochs
    for t in range(time):
        print("epoch",t,":")
        for i in range(swarm_size):
            #get the fitness of each net at this epoch
            particles[i].access_fitness(input_vector)
            if(particles[i].optima < global_optima.optima):
                #if we find aglobal optima, update it 
                global_optima = particles[i]
                print("particle ",i," has the global optima:",global_optima.optima)
        # update the velocities
        for j in range(swarm_size):
            particles[j].update_velocites( global_optima)
        #update the values of the neural nets   
        for k in range(swarm_size):
            particles[k].update_values()

    e = round(global_optima.net.error(global_optima.net.final_layer_outputs[0],input_vector)*100, 2)
    print("error =",e,"%\n")    

for i in range(1):
    main()