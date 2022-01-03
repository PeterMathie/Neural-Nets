from types import new_class
import numpy as np
from numpy.core.defchararray import array 

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
    W = np.random.uniform(low=-0.5, high=1.8, size=(x,))
    return W

#returns a random number between -1 and 1
def init_bias():
    b = np.random.uniform(low=-1.0, high=0.9)
    return b
#Activation_Functions class holds all relevant functions for the implemented activation functions
#neurons are made with an instance of Activation_Functions as a parameter
class Activation_Functions:

    #initialise a neuron's func but passing a string. The string is the name of the function
    def __init__(self, function):
        if(function=="sigmoid"):
            self.function = "sigmoid"
        if(function=="ReLU"):
            self.function = "ReLU"
        if(function=="tanh"):
            self.function = "tanh"
        if(function=="softPlus"):
            self.function = "softPlus"

    #nerons will call the activate function through their funciton parameter
    #This function acts as a hub where you can call the relevant activation function 
    def activate(self, linear_vector):
        if(self.function=="sigmoid"):
            return self.sigmoid(linear_vector)
        if(self.function=="ReLU"):
            return self.ReLU(linear_vector)
        if(self.function=="tanh"):
            return self.tanh(linear_vector)
        if(self.function=="softPlus"):
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
            if(z<-3):
                activation_vector.append(0)
            elif(z>3):
                activation_vector.append(1)
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

    def update(self, neuron, prev_layer_output, input_vector):
        if(self.function=="sigmoid"):
            self.update_weights_sigmoid(neuron, prev_layer_output, input_vector)
            self.update_bias_sigmoid(neuron, prev_layer_output, input_vector)

        if(self.function=="ReLU"):
            self.update_weights_ReLU(neuron, prev_layer_output, input_vector)
            self.update_bias_ReLU(neuron, prev_layer_output, input_vector)

        if(self.function=="tanh"):
            self.update_weights_tanh(neuron, prev_layer_output, input_vector)
            self.update_bias_tanh(neuron, prev_layer_output, input_vector)
        
        if(self.function=="softPlus"):
            self.update_weights_softPlus(neuron, prev_layer_output, input_vector)
            self.update_bias_softPlus(neuron, prev_layer_output, input_vector)

    #updates the weight values based on the weights equation
    #updates the weights for all values in the input data
    #partial differential of dJ(w,b)/dw = dz/dw * da/dz * dJ/da
    #one value for each weight vector
    def update_weights_sigmoid(self, neuron, prev_layer_output, input_vector):
        #the nxm dimensional gradient vector where n is the number of inputs and m is the number of weights for a given node
        dJ_dW = []
        for j in range(len(input_vector)):
            #change in the linear model output with respect to the weight
            #equivalent to the output from the previous layer of the net -> an array with as many elements as there are weights from the last layer to the current layer
            dz_dw = prev_layer_output[j]
            #change in the activation function output with respect to the linear model output
            #differential of the sigmoid function
            #neuron.output[j] is the jth output from the current neuron and its derived from the jth row in the input dataset
            da_dz = neuron.output[j] * (1-neuron.output[j])
            #the next function have the potential for a divid by 0, this prevents it by limiting the values
            if(neuron.output[j] == 1):
                neuron.output[j] = 0.999
            if(neuron.output[j] == 0):
                neuron.output[j] = 0.001
            #differential of the loss function wiht respect to the activation function
            dJ_da = (-input_vector[j][4]/neuron.output[j])+(1-input_vector[j][4])/(1-neuron.output[j])
            dJ_dw = []
            #the gradient for each weight, stored in the dJ_dw vector
            for i in range(0,(len(neuron.weights_vector)-1)):
                #dJ/dW = dz/dw * da/dz *dJ/da
                x = dz_dw[i]*da_dz*dJ_da
                dJ_dw.append(x)
            #the gradient vector for all inputs
            dJ_dW.append(dJ_dw)
        dJ_dW = np.transpose(dJ_dW)
        #updates the gradient
        for i in range(len(neuron.weights_vector)-1):
            mean=np.mean(dJ_dW[i])
            neuron.weights_vector[i] = neuron.weights_vector[i] - (neuron.learning_rate * mean)
        return neuron.weights_vector
    #works similar to the update_weights_sigmoid function
    def update_bias_sigmoid(self, neuron,  prev_layer_output, input_vector):
        Dj_db = []
        for j in range(len(input_vector)-1):
            #the differential of the linear fucntion with respect to the bias simplfies to 1
            dz_db = 1
            da_dz = neuron.output[j] * (1-neuron.output[j])
            if(neuron.output[j] == 1):
                neuron.output[j] = 0.999
            if(neuron.output[j] == 0):
                neuron.output[j] = 0.001
            dJ_da = (-input_vector[j][4]/neuron.output[j])+((1-input_vector[j][4])/(1-neuron.output[j]))
            x = (dz_db*da_dz*dJ_da)
            Dj_db.append(x)
        mean = np.mean(Dj_db)
        neuron.bias = neuron.bias - (neuron.learning_rate * mean)
        return neuron.bias

    def update_weights_ReLU(self, neuron,  prev_layer_output, input_vector):
        dJ_dW = []
        mean = []
        for j in range(len(input_vector)):
            dz_dw = prev_layer_output[j]
            #the differntial of the ReLU function is 0 for values less than 0 and 1 for all others
            if (neuron.output[j] <= 0.001):
                da_dz = 0
            elif (neuron.output[j] >= 0.999):
                da_dz = 0
            else:
                da_dz =  1
            if(neuron.output[j] == 1):
                neuron.output[j] = 0.999
            if(neuron.output[j] == 0):
                neuron.output[j] = 0.001
            dJ_da = (-input_vector[j][4]/neuron.output[j])+(1-input_vector[j][4])/(1-neuron.output[j])
            dJ_dw = []
            for i in range(0,(len(neuron.weights_vector)-1)):
                x = dz_dw[i]*da_dz*dJ_da
                dJ_dw.append(x)
            dJ_dW.append(dJ_dw)
        dJ_dW = np.transpose(dJ_dW)
        for i in range(0,len(neuron.weights_vector)-1):
            mean.append(np.mean(dJ_dW[i]))
            neuron.weights_vector[i] = neuron.weights_vector[i] - neuron.learning_rate * mean[i]
        return neuron.weights_vector

    def update_bias_ReLU(self, neuron,  prev_layer_output, input_vector):
        Dj_db = []
        for j in range(len(input_vector)-1):
            dz_db = 1
            da_dz = 0 if neuron.output[j] < 0 else 1
            dJ_da = (-input_vector[j][4]/neuron.output[j])+((1-input_vector[j][4])/(1-neuron.output[j]))
            x = (dz_db*da_dz*dJ_da)
            Dj_db.append(x)
        mean = np.mean(Dj_db)
        neuron.bias = neuron.bias - (neuron.learning_rate * mean)
        return neuron.bias

    def update_weights_tanh(self, neuron,  prev_layer_output, input_vector):
        #the n x m dimensional gradient vector where n is the number of inputs and m is the number of weights for a given node
        dJ_dW = []
        #The mean of each row in the dJ-Dw
        mean = []
        for j in range(len(input_vector)-1):
            dz_dw = prev_layer_output[j]
            #differential of the tanh fucntion 1/tanh(x)^2 where x is the output of the 
            da_dz = (1- np.power(np.tanh(neuron.output[j]),2))
            if(neuron.output[j] == 1):
                neuron.output[j] = 0.999
            if(neuron.output[j] == 0):
                neuron.output[j] = 0.0001
            dJ_da = (-input_vector[j][4]/neuron.output[j])+(1-input_vector[j][4])/(1-neuron.output[j])
            dJ_dw = []
            for i in range(0,(len(neuron.weights_vector)-1)):
                x = dz_dw[i]*da_dz*dJ_da
                dJ_dw.append(x)
            dJ_dW.append(dJ_dw)
        dJ_dW = np.transpose(dJ_dW)
        for i in range(0,len(neuron.weights_vector)-1):
            mean.append(np.mean(dJ_dW[i]))
            neuron.weights_vector[i] = neuron.weights_vector[i] - (neuron.learning_rate * mean[i])
        return neuron.weights_vector

    def update_bias_tanh(self, neuron,  prev_layer_output, input_vector):
        Dj_db = []
        for j in range(len(input_vector)-1):
            dz_db = 1
            da_dz = (1- np.power(np.tanh(neuron.output[j]),2))
            dJ_da = (-input_vector[j][4]/neuron.output[j])+((1-input_vector[j][4])/(1-neuron.output[j]))
            x = (dz_db*da_dz*dJ_da)
            Dj_db.append(x)
        mean = np.mean(Dj_db)
        neuron.bias = neuron.bias - (neuron.learning_rate * mean)
        return neuron.bias 

    def update_weights_softPlus(self, neuron,  prev_layer_output, input_vector):
        #the n x m dimensional gradient vector where n is the number of inputs and m is the number of weights for a given node
        dJ_dW = []
        #The mean of each row in the dJ-Dw
        mean = []
        for j in range(len(input_vector)-1):
            dz_dw = prev_layer_output[j]
            da_dz = np.power(1+np.power(2.71828,neuron.output[j]),-1)
            if(neuron.output[j] == 1):
                neuron.output[j] = 0.999
            if(neuron.output[j] == 0):
                neuron.output[j] = 0.001
            dJ_da = (-input_vector[j][4]/neuron.output[j])+(1-input_vector[j][4])/(1-neuron.output[j])
            dJ_dw = []
            for i in range(0,(len(neuron.weights_vector)-1)):
                x = dz_dw[i]*da_dz*dJ_da
                dJ_dw.append(x)
            dJ_dW.append(dJ_dw)
        dJ_dW = np.transpose(dJ_dW)
        
        for i in range(0,len(neuron.weights_vector)-1):
            mean.append(np.mean(dJ_dW[i]))
            neuron.weights_vector[i] = neuron.weights_vector[i] - (neuron.learning_rate * mean[i])
        return neuron.weights_vector

    def update_bias_softPlus(self, neuron,  prev_layer_output, input_vector):
        Dj_db = []
        for j in range(len(input_vector)-1):
            dz_db = 1
            da_dz = np.power(1+np.power(2.71828,neuron.output[j]),-1)
            dJ_da = (-input_vector[j][4]/neuron.output[j])+((1-input_vector[j][4])/(1-neuron.output[j]))
            x = (dz_db*da_dz*dJ_da)
            Dj_db.append(x)
        mean = np.mean(Dj_db)
        neuron.bias = neuron.bias - neuron.learning_rate * mean
        return neuron.bias 

#Neuron class defines the neuron's hyperparameters and the linear model
class Neuron:
    def __init__(self, weights_vector, bias, learning_rate, activation_func, layer):
        # a vector containing all the weights of the neuron
        self.weights_vector = np.array(weights_vector)
        # the neuron's bias
        self.bias = bias
        # the learning rate of the neuron (could have been a net param as it is the same for all neurons)
        self.learning_rate = learning_rate
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
            linear_vector = np.matmul(input_vector[::],weights_tansposed)
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
        # the learning rate to be used for each neuron
        learning_rate = 0.05
        # define sizes of each row as arrays that will store neurons 
        input_layer = np.empty(dtype=Neuron, shape=(self.nodes_in))
        hidden_layers = self.init_hidden_structure(self.nodes_hidden) 
        output_layer = np.empty(dtype=Neuron, shape=(self.nodes_out))

        # for make nodes_in many neurons and add them to the input_layer array
        for i in range(self.nodes_in):
            weights_vector = np.array(init_weights(self.num_inputs))
            bias = init_bias()
            input_layer[i] = Neuron(weights_vector, bias, learning_rate, self.first_function, 0)

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
                hidden_layers[j][k] = Neuron(weights_vector, bias, learning_rate, self.hidden_functions[j], j+1)

        # make the node_out many nodes for the output layer
        for l in range(self.nodes_out):
            #as many wieghts as nodes in one hidden layer
            weights_vector = np.array(init_weights(self.nodes_hidden[j]))
            bias = init_bias()
            output_layer[l] = Neuron(weights_vector, bias, learning_rate, self.out_function,(j+2))
        
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
    
    #updates the weights for each node in the final layer
    def back_propagation_final(self, input_vector):  
        prev_layer_output = self.hidden_layers_outputs[len(self.hidden_layers)-1]
        for node in self.output_layer:
            node.activation_func.update(node, prev_layer_output, input_vector)

    def back_propagation_hidden(self, input_vector):   
        # work backwards through the list of hidden vectors
        for layer in reversed(self.hidden_layers):
            for node in layer:
                if (node.layer==1):
                    #the previous output for the first hidden layer is the first layer's
                    prev_layer_output = self.first_layer_outputs
                else:
                    #get the previous layers output
                    prev_layer_output = self.hidden_layers_outputs[node.layer-1]
                node.activation_func.update(node, prev_layer_output, input_vector)

    def back_propagation_first(self, input_vector):   
        for node in self.input_layer:
            node.activation_func.update(node, input_vector, input_vector)
 

    def run(self,input_vector):
        jold = 100
        j = 10
        i=0
        #when the change in cost function is very small, we have found the ~turning point
        while( i <= 1000):   

            #iterate through the layers and populate values for the activation functions
            #output from one function is the input to the next
            activation_vector = self.forward_pass_first(input_vector)
            activation_vector = self.forward_pass_hidden(activation_vector)
            activation_vector = self.forward_pass_final(activation_vector)

            #find the cross entropy loss
            loss_vector = self.loss(activation_vector,input_vector)

            #print("\n-------ACTIVATION VECTOR-------\n",np.array(activation_vector))           
            #print("\n-------LOSS VECTOR-------\n",np.array(loss_vector))
            
            #perform back propagation to update the heweights and biases
            self.back_propagation_final(input_vector)   
            self.back_propagation_hidden(input_vector)   
            self.back_propagation_first(input_vector)   
            print(i)
            #store the old cost in jold
            jold = j  
            #measure the new cost in j, these will be used to measure the change in cost from 
            j = self.cost(loss_vector)
            i=i+1
        #the error for printing
        e = round(self.error(activation_vector,input_vector)*100, 2)
        #print("\n-------ACTIVATION VECTOR-------\n",activation_vector)
        #print("\n-------LOSS VECTOR-------\n",np.array(loss_vector))
        print(j)
        print("\nerror =",e,"%\n")     

def main( ):
    #HYPERPARAMETERS 
    #the input vector from our dataset
    input_vector = np.array(get_inputs())
    #the number of attributes in out dataset
    num_inputs = len(input_vector[0])-1

    #the nodes for the first layre
    nodes_in = 2
    #the nodes for the hidden layers, each layer can have any number of nodes
    nodes_hidden = [2]
    #nodes for the output layer (has to be 1)
    nodes_out = 1

    #number of hidden layers we want
    num_hidden_layers = len(nodes_hidden)

    #activation functions for thier respective layers
    first_function = "sigmoid"
    hidden_functions = ["ReLU","sigmoid","ReLU","sigmoid","ReLU","sigmoid","ReLU","sigmoid"]
    out_function = "sigmoid"
    functions = [first_function,hidden_functions,out_function]


    net = Net(num_inputs, num_hidden_layers, nodes_in, nodes_hidden, nodes_out, functions)

    net.run(input_vector)

    #for node in net.input_layer:
    #    print("\n",node.weights_vector)
    #for layer in net.hidden_layers:
    #    for node in layer:
    #        print("\n",node.weights_vector)
    #for node in net.output_layer:
    #    print("\n",node.weights_vector)


for i in range(1):
    main()