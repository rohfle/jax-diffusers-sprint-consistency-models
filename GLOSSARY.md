GLOSSARY OF TERMS

### what is convolution
- a mixture of two functions
- input + convolutional filter => training weights
- kernel applied in sliding fashion to rows and columns
- kind of like gaussian blur?
- reduces number of weights required
### what is convolutional layer
- https://developers.google.com/machine-learning/glossary/#convolutional-layer
### what is pooling layer
- reducing a matrix created by a previous layer to a smaller matrix
### what is dense layer
- a layer where every node is connected to every node in subsequent layer
### what is depth
- sum of hidden layer count, output layer count, embedding layer count in neural network
### what is dropout
- removes a random selection of fixed number of units in network layer for a single step, helps prevent overfitting
### what is rank
- the number of dimensions in a tensor
- scalar = 0, vector = 1, matrix = 2
### what is a feature
- an input part of an example for a machine learning model
### what is a label
- the result part of an example for machine learning model
### what is a hyperparameter
- things adjusted by user or tuning service during successive runs of training a model
- eg learning rate
### what is fine tuning
- optimising already trained model using new training data to fit a new problem
### what is an embedding layer
- a special hidden layer that trains on a high-dimensional categorical feature to gradually learn a lower dimension embedding vector
- eg a 73000 one-hot encoding vector to a 12 dimension vector with real numbers
- to do with speed and not multiplying 72999 zeroes
### what is an encoder
- raw, sparse representation to more processed, denser representation
### what is attention
- any neural network mechanisms that aggregate information from a set of inputs
- eg weighted sum of inputs where weight is computed from another part of the neural network
### what is self attention
- transforms a sequence of embeddings into another sequence of embeddings
### what is activation function
- enables neural networks to learn nonlinear relationships
- eg sigmoid, relu
### what is softmax
- makes probabilities of each class in multiclass classification model equal to one
### what is a timestep

### what is l1 loss
### what is l1 regularization
### what is l2 loss
- MSE = l2 loss / item count in batch
### what is l2 regularization

### what is a step
- a forward and backward pass of one batch, similar to iteration
### what is an iteration
- a single update of the models parameters
### what is a schedule
- eg noise schedule sigma_karras
### what is distilling
- aims to create smaller and more efficient models while maintaining accuracy
### what is one-shot learning
- learn effective classifiers from a single example
### what is a recurrent neural network
- a neural network that is intentionally run multiple times, where parts of each run feed into the next run. Specifically, hidden layers from the previous run provide part of the input to the same hidden layer in the next run. Recurrent neural networks are particularly useful for evaluating sequences, so that the hidden layers can learn from previous runs of the neural network on earlier parts of the sequence.
### what is a model
- input + params => output
### what is normalization
- the process of converting actual range of values to a standard range of values such as -1 to 1, 0 to 1 etc
### what is learning rate
- tells grads how strong to adjust params
### what is forward pass
- model(params, inputs) => preds
- loss = sum(abs(preds - expecteds))
- capture grads for backprop
### what is backpropagation
- use grads and params and learning rate => new params
### what is batch normalization
- normalizing input and output of activation functions in hidden layer
- reduces overfitting, adds stability by protecting from outlier weights, enables higher learning rates
### what is a bayesian neural network
- predicts a distribution of values instead of a single value
- eg standard house price predicted $900,000 vs bayesian $900,000 with sd of 67,200
### what is denoising
- noise is artificially added to dataset
- the model tries to remove the noise