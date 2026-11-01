
## Abstract

We train **Multi Layers Perceptron** (MLP) and **Convolutional Neural Networks** (CNN) to recognize handwritten digit and letter in Extended MNIST dataset with 47 labels, the first 10 labels are number from 0 to 9, others are letters (letters have the same style in both capital and non-capital will be considered as the capital one). We achieved the accuracy of over 88% on the test data using MLP and approximately 90% using CNN with the help of TensorFlow.

## Quick start
- Require configuration : RAM > 8GB
- Library required : TensorFlow, numpy, pandas, matplotlib, sklearn
- Kaggle's dataset: [[Train](https://www.kaggle.com/crawford/emnist?select=emnist-bymerge-train.csv), [Test](https://www.kaggle.com/crawford/emnist?select=emnist-bymerge-test.csv)]
- There are 2 ways to run code: Google colab and not Google colab
    - For Colab user: Create a Kaggle account (if not have any), then go to account settings and click the "Create New API Token" as picture below.
     ![](https://i.imgur.com/uHM7dmo.jpg)
        - A json file named kaggle will be downloaded, drag it into the folder space.
        - Run the code below to finish the preparation.
    ```
    !pip install -q kaggle 
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !ls ~/.kaggle
    !chmod 600 /root/.kaggle/kaggle.json
    !kaggle datasets download -d crawford/emnist
    !unzip emnist.zip emnist-balanced-test.csv
    !unzip emnist.zip emnist-balanced-train.csv
    !unzip emnist.zip emnist-balanced-mapping.txt
    ```
    - Other user:
        - Download Kaggle's data from the link above
        - Install requirements
        - Run notebook
- If you can not train the model, we trained them and dumped to `.h5` file in this. You can load them by running and test the last two cells. Remember to read test data first, or else you will not have any data to test.

## Introduction
- Since the beginning, Image Recognition (IR), a subcategory of Computer Vision and Artificial Intelligence, has brought to daylife problems many solutions. This process can be applied to many different specialities such as medical to diagnose diseases though image samples, traffic to detect violation, security to distinguish a person to another,... So it is not overstated to said that IR is one of the hottest topics of Computer Science study.
- To archive these goal, our main distribution here is looking into a few mainstream image recognition models on EMNIST data set. Among of them, there are MLP and CNN respectively. These two are the most fundamental models, which were proved to have good performance in IR. We implement both to have a better view about their structure as well as appoach many other terminologies involved in Machine Learning.
- Ultimately, we bring out the conclusion and acknowledgement in the Conclusion section.

## Background and Related Work
### EMNIST
* The EMNIST dataset is a set of handwritten character digits derived from the [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19), which contains digits, uppercase and lowercase handwritten letters, and converted to a 28x28 pixel image format and dataset structure that directly matches the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). This dataset is more challenging for recognized models because of the variations of data such as rotate, flip, blur,... But it will also increase the model ability to detect writting which is not in formal form as well as the accuracy.
* For dataset, we use **Kaggle's** By_Merge Extended MNIST which contains 814255 handwritten digits and letters (697931 for training and 116324 for testing). Dataset is stored in *.csv* file, which first column is the label of the image, other columns are the pixels of an 28x28 image, which means 784 columns, while each row is a sample image. There are total 47 labels, corresponding as Figure 1 below:
![47 labels](https://i.imgur.com/xBW35pu.png)
<p>
    <center><em>Figure 1: Visual breakdown of By_Merge EMNIST dataset [1] </em></center>
</p>

<center><img src="https://i.imgur.com/mT7IYXQ.png" alt="dummy_data" width="60%"/></center>
<p>
    <center><em>Figure 2: By_Merge EMNIST samples </em></center>
</p>

### Artificial Neural Network (ANN)
- The concept of ANN is based on the subject of biology where neuron network plays a important role in human body. The whole process of receiving and sending signals is done in particular manner like a neuron receive signals from  other neuron through dendrites. The Neuron sends signals at  spikes of electrical activity through an axon and an axon splits these signals through synapse and send it to the other neurons. [26]
<center><img src="https://i.imgur.com/HiXIYQj.png" alt="biology_NN" width="60%"/></center>
<p>
    <center><em>Figure 3: Biological neuron </em></center>
</p>

- Similar to biological neuron, an artificial neuron has a number of input channels, a processing stage, and one output that can fan out to multiple other artificial neurons. A processing stage may include a large amount of simple processing element that are interconnected to each other and most of the time, it receives arguments such as weights, inputs,... and derives output by passing them to the activate function.
<center><img src="https://i.imgur.com/ie4LqN4.png" alt="biology_NN" width="70%"/></center>
<p>
    <center><em>Figure 4: Artificial neuron (Perceptron) </em></center>
</p>

- By stacking many artifical neurons, we form a layer, put these layers together, a multilayer neuron network is created. The most basic example is what we are going to discuss below.

### Convolutional Neural Network (CNN) 
- As far as we know, CNN is the best image recognition model. The problem of other fully connected neural network  model is mostly because of image processing. If the image is large which means the number of pixels is also very large, input layer may have $n^2$ units (square image), with weights and hidden layers, parameters may increase rapidly $\rightarrow$ With convolution operation, we can solve this problem but still can extract image's features.
- Beside the basic fully connected neuron network layers, CNN have some convolutional layers before processing. Basic structure is
$$\text{Input image}\rightarrow \text{Convolutional layer (Conv)} + \text{Pooling layer (Pool)} \rightarrow \text{Fully connected layer (FC)} \rightarrow \text{Output}$$
- There are several architectures in the field of Convolutional Networks that have a name. The most common are [27]:
    - LeNet
    - AlexNet
    - GoogleNet
    - VGG Net
    - Res Net
- In this project, we will not dig to deep in CNN variations but only look into traditional CNN.

## Approach
### 0. Load dataset
- As mentioned above, we use Kaggle's data set as csv form. By using `pandas` to read data, we can easily get the description of each sample without having to flatten it
- Furthermore, our dataset has letters and digits, so labels will receive a label in range 0 - 46. In Application Mathematics, we had a chance to implement one-hot encoding to represent labels and using `softmax`, `argmax` to extract the right label. For easier approach, we use this method to display value of labels. Generally speaking, we convert an integer value in the range to an array of length 47, at every index is the probality that the label will be received. When compiling the model, this solution leads to an argument passed to the function which is `loss=categorical_crossentropy`
Example: Label = 5 $\rightarrow$ Array: [0 0 0 0 1 0 ... 0] 
- Data is represented as gray images so the value of each pixel is in the range 0 - 255, to normalize, we divide them to 255, minimize the value for computing.

### 1. Multi-layer Perceptron
#### 1.1. Structure
- Input layer's shape will be ((784,)): which is a 784 units.
- Both hidden layers will have 512 units, both activate functions are **ReLU**
- We initalize the weight and biases for each hidden layer, Since we did not specify, so by default [13]:
    - Weight (Kernel): Glorot Uniform, which will draw a sample $[-limit, limit]$
     where:
$$limit = \sqrt{\frac{6}{(input + output)}}$$
        - input: the number of input units in the weight tensor
        - output: the number of output units. 
    - Biases: Zero
- Output layer contains 47 units, which is 47 labels, and the activate function is **softmax**.
    ![](https://i.imgur.com/YN4q6YZ.png)
    <p>
    <center><em>Figure 5: Multi-layer Perceptron Visualization</em></center>
    </p>

#### 1.2. Hyperparameters
- We trained our model using Adam optimizer [14] to update the weights between each layer with a minibatch of 256 images, which will have its parameters by default
    - Learning rate: **0.001**.
    - $\beta_1$ = 0.9
    - $\beta_2$ = 0.999
    - $\epsilon$ = 1e-07
    - $amsgrad$ = False
    - $name$ = "Adam"
- 2 Dropout layers with a parameter will take a rate, which will reduce the number of output after it pass the activate functions to prevent overfitting. 
    - In this project, we use the rate $\alpha = 0.2$ which is 20% of output will be dropout.

#### 1.3 Processing
- Transform Label: by convert a class vector, which is train label and test_label to the binary class matrix with 47 different labels.
- Train model: 
    - We will fit it with epochs = 10, using a batch_size of 256 images
    - Using Adam optimizer to optimize the weight, with the loss is <b>categorical crossentropy</b> and the metrics will take <b>accuracy</b> as the measure.
        - First, it takes inputs and pass the first Dense Layer. Here, it will calculate the output by this formula:
    $$output_{h_1} = activation(dot(input, kernel) + bias)$$
        - After that, it will pass the Dropout layer to drop out 20% of the output after the activation that would the incorrect result.
        - It will do the same when it pass the second Dense Layer and Dropout Layer
    $$output_{h_2} = activation(dot(output_{h_1}, kernel) + bias)$$
        - Finally, do the **softmax** at the Output layer.

### 2. Convolutional Neural Network
#### 2.1. Structure
- There are 2 Convolutional Blocks
- Each block contains 2 **Conv2D** layers with **LeakyReLU** activation layers with $$\alpha = 0.3$$ by default. LeakyReLU has the formula as below.
- Then a **MaxPool2D** layer and finally a **Dropout Layer**.
- Then Dense Layers and Output layer after **Flatten layer**.
**Dropout layer** drops the few activation nodes while training, which acts as regularization. Do not let the model to over-fit.
- There are 3 Dense layers, 2 first layers has the **ReLU** activation function. 
- Output layer has 47 nodes with **softmax** activation.
![](https://i.imgur.com/GLHUejv.jpg)
    <p>
    <center><em>Figure 6: Illustration of Convolutional Neural Networks by Tarun Kumar [8]</em></center>
    </p>
    
#### 2.2. Hyperparameters
- We using Adam optimizer, with same parameters as MLP model above.

#### 2.3. Processing
- All train and test datas are now reshaped as $[28, 28, 1]$ like images
- The first convolutional layers will use 32 filters with the size  $3 \times 3$.
- After convolution, images can be downsize, so we have to zero pad the picture in order to preserve the size with parameter ``padding="same"`` 
- In the convolve step, filter will be stack up on the image matrix and multiply with part of it. Keep striding the filter accross the height and width of image and multiplying. Result is the 28x28 convolved matrix of image. For 32 filters we have 32 different matrices so the image now will have the shape $[28, 28, 32]$.
![](https://i.imgur.com/DgsENnZ.jpg)
<p>
    <center><em>Figure 7: Illustration of convolve image [28]</em></center>
    </p>

- After the convolution, LeakyReLU function is used to calculate output.
- MaxPool2D layer is used to reduce the size of the image, it use a window with size
$$(2,2)$$
to reducing the image from $$(28,28)$$ to 
$$(14,14)$$
by choosing the max value of each $[2,2]$ part of image and put it into new max-pooled matrix.
![](https://i.imgur.com/8llgrwZ.jpg)
<p>
    <center><em>Figure 8: Illustration of max-pooling. [27]</em></center>
    </p>

- Repeat the process above one more time.
- Block 2 is same as block 1, but number of filters changed from 32 to 64.
- After convolution and max pooling, data will be flatten into 49 units because image after 2 times of max-pooling will be downsized to $7\times 7$.
- Dense layers is used to calculate the output of each units with the activation given.

### 3. Hyperparameters and Regularization
#### 3.1 Dropout
- In neural networks the regularization technique used to reduce overfitting by preventing co-adaptations on training datais dropout. While training neural network the technique dropout is used which randomly dropping out the neurons in the  learning  stage. After a layer in MLP or convolutional block in CNN, dropout is introduced in this architecture to  reduce overfitting problem. [29]
- When we use fully connected NN, neurons are fully dependent. This technique will force the model to find robust features with a $p$ probability to discard the good neuron to train others. 
- In our models, we pick p in range 0.2 - 0.25, which means 20% to 25% of neuron will be droped out.

#### 3.2. Batch size, number of epochs and learning rate
- Batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters. If we split data to many batches, we may save some memories. Also ConvNet is sensitive to batch size. Here we choose batch size is 256
- Number of epochs is the number of complete pass through the entire training set. Here we choose `epochs=10` for each model. The reason why this argument is small is that the data set is large so it will take less epoch to reach the best accuracy rate. Also, if number of epochs is too big, model might be overtrained.
- Learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function [30]. To choose the best learning rate we have to try every value and choose the best one. Here we choose the default learning rate of Adam optimizer. In some datasets/problems, this value is not a good one but because of the lack of time and hardware, we are unable to go through the whole process.

#### 3.3. Activate function
- ReLU:
    - As is shown in [Convolution Neural Networks - Jianxin Wu](https://cs.nju.edu.cn/wujx/teaching/15_CNN.pdf), we use a symbol shown in boldface to represent a vector - e.g.,$x∈R^D$ is a column vector with D elements. We use a capital letter to denote a matrix—e.g.,$X∈R^{H×W}$ is a matrix with H rows and W columns.  The vector x can also be viewed as a matrix with 1 column and D rows. [25]
    $$f(x) = x^+ = max(0, x^l_{i, j, d})$$
    where $0 \leq i < H^l = H^{l+1}$, $0 \leq i < W^l = W^{l+1}$, $0 \leq i < D^l = D^{l+1}$ [25]
    Hence, we have:
    $$[\frac{\delta z}{\delta x^l}]_{i, j, d} = \begin{cases}
    [\frac{\delta z}{\delta y}]_{i, j, d} \ \text{if} \ x^l_{i, j, d} \\ 0 \ \text{otherwise} \end{cases}$$
    ![](https://i.imgur.com/2j9x2J5.png)
    <p>
    <center><em>Figure 7: The ReLU function [25]</em></center>
    </p>

- Softmax: Normalize the output of a network to a probability distribution over predicted output classes.
Formula:
    $$\sigma(\vec{z})_i = \frac{e^{z_i}}{\Sigma^K_{j=1}e^{z_j}} $$
    with:
        - $\vec{z}$: The input vector to the softmax function, made up of $z_i = 1, 2, ..., K$
        - $z_i$: elements of the input vector to the softmax function, and they can take any real value, positive, zero or negative.
        - $e^{z_i}$: The standard exponential function is applied to each element of the input vector. This gives a positive value above 0, which will be very small if the input was negative, and very large if the input was large. However, it is still not fixed in the range (0, 1) which is what is required of a probability.
        - $\Sigma^K_{j=1}e^{z_j}$: The term on the bottom of the formula is the normalization term. It ensures that all the output values of the function will sum to 1 and each be in the range (0, 1), thus constituting a valid probability distribution.
        - $K$: Number of classes in multi-class classifier
- Leaky ReLU: A variation of ReLU to solve dead neuron problem which happens when your ReLU always have values under 0 - this completely blocks learning in the ReLU because of gradients of 0 in the negative part
    $$y_i = \begin{cases} x_i \ \text{if} \ x_i \geq 0 \\ \frac{x_i}{a_i} \ \text{if} \ x_i < 0 \end{cases}$$

#### 3.4. Optimizer
##### 3.4.1. Stochastic Gradient Descent
- Basic algorithm to help optimize the neural network, make neural networks converge
- Stochastic Gradient Descent will only calculate the cost of one example in each step.
- SGD formula is used to update the parameter in the backward pass, specifically is the weight of each layer, using backpropagation (using in almost optimizer for update the weight) to calculate the gradient:
    $$\theta = \theta - \eta * \nabla_\theta J(\theta; x, y)$$
    - $\theta$, is the weights of neural networks. We here to update its weight to get the correct the satisfied result.
    - Learning rate $\eta$
    - $J$ is formally known as objective function, but most often it's called cost function or loss function. 
- You can have a better a visualization within the [Optimizers Explained](https://mlfromscratch.com/optimizers-explained/#stochastic-gradient-descent), where it will show a gif that demonstrate how SGD work, and [Gradient descent, how neural networks learn | Deep learning, chapter 2 - 3Blues1Brown](https://www.youtube.com/watch?v=IHZwWFHWa-w) will show you how Neural Networks learn.

    ![](https://i.imgur.com/a5LwAhI.png)
    <p>
    <center><em>Figure 8: Example of SGD after it reach local minima [24]</em></center>
    </p>
- Pros and Cons:
    - Pros:
        - Faster compared to the Normal Gradient Descent approach
    - Cons:
        - Converges slower compared to newer algorithms
        - Has more problems with being stuck in a local minimum than newer approaches.
        
##### 3.4.2. Adam
Let us have a glance at Adam:
- Adam is an adaptive learning rate method, which means, it computes individual learning rates for different parameters. Its name is derived from adaptive moment estimation, and the reason it’s called that is because Adam uses estimations of **first** and **second moments** of gradient to adapt the learning rate for each weight of the neural network - from [4]
- It combines advantages of two extensions stochastic gradient descent to observe the learning rate. Specifically:
    - Stochastic Gradient Descent with Momentum, specifically  is AdaGrad
    - Root Mean Square Propagation (RMSProp)
- The details of the two extensions is in [Optimizers Explained](https://mlfromscratch.com/optimizers-explained/#adam)
- The caculation is followed by these calculation to update the all the weights.
    $$w_t = w_{t-1} - \eta\frac{\hat{n}}{\hat{v} + \epsilon}$$
    where
    $$\hat{m} = \frac{m_t}{1-\beta^t_1}$$
    $$\hat{v} = \frac{v_t}{1-\beta^t_2}$$
    where
    $$m_t = \beta_1m_{t-1}+(1-\beta_1)g_t$$
    $$v_t = \beta_2v_{t-1}+(1-\beta_2)g_t$$
    - Epsilon $\epsilon$, which is just a small term preventing division by zero. This term is usually $10^{-8}$
    - Learning rate $\eta$ (although it's $\alpha$) in the paper. They explain that a good default setting is $\eta = 0.001$, which is also the default learning rate in **TensorFlow Keras**.
    - The gradient g, which is still the same thing as before: $g = \nabla J(\theta_t,i)$

##### Conclusion
The reason we chose Adam as our optimizer because:
- It converges faster, which will lower the cost.
- The result is consistency, based on the implement, the accuracy is outstanding.

## Experiment
- In this section, we will show you the result of training and testing each model with the parameters we set above.
- We use By_Merge EMNIST [1] dataset to train both model and evaluate the accuracy and the loss.
    - Firstly, we split the training set into training and validation set to train and evaluate the model with the percentage of 90% - 10%
    - Secondly, we plot the graph of accuracy and loss in training process to detect if the model is overfitted or not.
    - Thirdly, we predict the test set with the above model, generate confusion matrix, display some samples as well as their labels and predictions. To be more specific, we will show you 100 samples that the model predicted wrong and explain why
- Multilayer Perceptron:
    - Accuracy and loss graph:
    ![](https://i.imgur.com/HGuxOvw.png)
    ![](https://i.imgur.com/PI9IqbM.png)
    - Confusion matrix
    ![](https://i.imgur.com/DN2DIbf.png)
    - Testing result
    ![](https://i.imgur.com/z8TR8Nv.png)
    - Sample results
    ![](https://i.imgur.com/5YhCWD8.png)
    - Some error labeled samples:
    ![](https://i.imgur.com/yhNbHsT.png)
    - Conclusion: 
        - You can see that the letter 'O' and the number 0 can be easily predicted wrong, the same thing happens to 'Z' and 2, 'S' and 5,...
        - We did not re-rotate or flip data back to its right direction, which may lead to error in recognition
- Convolutional Neuron Network
    - Accuracy and loss graph:
    ![](https://i.imgur.com/UcRp3Is.png)
    ![](https://i.imgur.com/lXHk5md.png)
    - Confusion matrix
    ![](https://i.imgur.com/pzErGNu.png)
    - Sample results
    ![](https://i.imgur.com/qSV0xVc.png)
    - Some error labeled samples:
    ![](https://i.imgur.com/YHlsIRX.png)
    - Conclusion: 
        - Error in predictions are pretty much the same as MLP
        - CNN extract features through convolution layer, so if this layer is not configured optimally, it may lead to the result that the model can not detect that picture
        - Image processing also affects CNN performance

## Conclusion
### About models
- MLP trains faster because it do not need to pass data through convolution layers
- CNN accuracy increases more rapidly but can easily be overfitted
- Image preprocessing affects model's predictions
- Hyperparameters play an important role in building models. If there are more research time, we may get to know them better

### Acknowledgement
- How dataset was created based on different situations.
- Learn the way neural network recognize handwritten datas.
- How backpropagation process in neural network.
- There are different activate functions beside Sigmoid function, which have more benefits, for example : ReLU, Softmax,..
- Learn about the differences between Normal Gradient Descent and Stochastic Gradient Descent as well as other optimal functions

## References
* [1] [EMNIST: an extension of MNIST to handwrittenletters - Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre van Schaik](https://arxiv.org/pdf/1702.05373.pdf)
* [2] [Gradient Descent - StatQuest with Josh Starmer](https://www.youtube.com/watch?v=sDv4f4s2SB8)
* [3] [Sochastic Gradient Descent - StatQuest with Josh Starmer](https://www.youtube.com/watch?v=vMh0zPT0tLI&t=28s)
* [4] [Adam Algorithm - Vitaly Bushaev](https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c)
* [5] [Neural networks - 3Blue1Brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [6] [Multi-layer Perceptron and Backpropagation](https://machinelearningcoban.com/2017/02/24/mlp/)
* [7] [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com)
* [8] [Digit Recognition CNN tutorial - Tarun Kumar](https://www.kaggle.com/tarunkr/digit-recognition-tutorial-cnn-99-67-accuracy)
* [9] [Multi-layer Perceptron - Michael Nielson, Jerry Gagelman, Jeremy Vonderfecht](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py)
* [10] [Basic ANN Model with TensorFlow - Computer Science](https://www.youtube.com/watch?v=kOFUQB7u5Ck&list=PL4tzOw8T8xkSgw8fEF7sFn0ux0f8omm_v)
* [11] [Reference paper format](https://share.cocalc.com/share/32b94ee413d02759d719862907bb0a85a76c43f1/2016-11-07-175929.pdf)
* [12] [Layer weight initializer](https://keras.io/api/layers/initializers/)
* [13] [Adam Optimizer in Keras](https://keras.io/api/optimizers/adam/)
* [14] [Overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent)
* [15] [Overfitting](https://elitedatascience.com/overfitting-in-machine-learning)
* [16] [Adam Optimizer for Deep Learning](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
* [17] [Rectifier Neural Network (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
* [18] [Keras Model Training API](https://keras.io/api/models/model_training_apis/)
* [19] [Loss Function](https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class)
* [20] [Parameter and Hyperparameter](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)
* [21] [Valid and Same padding](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)
* [22] [Different kinds of convolutional filters](https://www.saama.com/different-kinds-convolutional-filters/)
* [23] [Rectified Linear Units](https://arxiv.org/pdf/1505.00853.pdf)
* [24] [Optimizers Explained](https://mlfromscratch.com/optimizers-explained/#/)
* [25] [Convolutional Neural Networks - Jianxin Wu](https://cs.nju.edu.cn/wujx/teaching/15_CNN.pdf)
* [26] [A Comprehensive Study of Artificial Neural Networks - Vidushi Sharma, Sachin Rai, Anurag Dev](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.9353&rep=rep1&type=pdf)
* [27] [CS231n: Convolutional Neural Networks for Visual Recognition - Stanford](https://cs231n.github.io/convolutional-networks/)
* [28] [Keras Conv2D and Convolutional Layers](https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/)
* [29] [Hyperparameter Optimization and Regularization on Fashion MNIST Classification](https://www.researchgate.net/publication/334947180_Hyperparameter_Optimization_and_Regularization_on_Fashion-MNIST_Classification)
* [30] [Learning rate Wiki](https://en.wikipedia.org/wiki/Learning_rate)