# Deep-Learning-A-Z

<img src="http://uc-r.github.io/public/images/analytics/deep_learning/deep_nn.png" alt="22" style="width:104px;height:142px;">

## Notes from the course Deep Learning A-Z provided by Udemy.

**SECTION 1 - Artificial Neural Networks (ANN)**:

Artificial neural networks (ANN) or connectionist systems are computing systems vaguely inspired by the biological neural networks that constitute animal brains.[1] The neural network itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs.Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it. (source: <a href="https://en.wikipedia.org/wiki/Artificial_neural_network"> wikipedia </a>) 

**Example:**

1) **Bank Customers**

    **Task 1 - Create the Artificial Neural Network**
    
    Predict which customers are most likely to leave a specific bank(binary classification problem). 
    For this task we used an Artificial Neural  Network with 11 feautures and 1 output Layer. The final accuracy of the model was 86.25%, without any evaluation technique(and thus not that relevant to the reality as the model needs tuning). 
    The code can be found <a href="https://github.com/gpsyrou/Deep-Learning-A-Z/blob/master/Artificial%20Neural%20Networks/bank_customers_pred.py"> here </a>.
        
    **Task 2 - Evaluate,Improve and Tune the ANN**
      
    Tuning and improving the ANN constructed in Task 1.
  
    **Readings**
    <ul>
    1) Yann LeCun et al., 1998, <em><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">Efficient BackProp</a></em> <br />
    2) By Xavier Glorot et al., 2011,&nbsp;<a href="http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf"><em>Deep sparse rectifier neural networks</em></a><br />
    3) CrossValidated, 2015,&nbsp;<a href="http://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications"><em>A list of cost functions used in neural networks, alongside applications</em></a><br />
    4) Andrew Trask, 2015,&nbsp;<a href="https://iamtrask.github.io/2015/07/27/python-network-part2/"><i>A Neural Network in 13 lines of Python (Part 2 – Gradient Descent)</i></a><br />
    5) Michael Nielsen, 2015,&nbsp;<a href="http://neuralnetworksanddeeplearning.com/chap2.html"><i>Neural Networks and Deep Learning</i></a><br />
    </ul>
    
    
<br></br>
**SECTION 2 - Convolutional Neural Networks (CNN)**
   
 <img src="https://cdn-images-1.medium.com/max/1600/1*NQQiyYqJJj4PSYAeWvxutg.png" alt="22" style="width:104px;height:142px;">
  
 CNNs use a variation of multilayer perceptrons designed to require minimal preprocessing.They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics.

Convolutional networks were inspired by biological processes in that the connectivity pattern between neurons resembles the organization of the animal visual cortex. Individual cortical neurons respond to stimuli only in a restricted region of the visual field known as the receptive field. The receptive fields of different neurons partially overlap such that they cover the entire visual field.

CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns the filters that in traditional algorithms were hand-engineered. This independence from prior knowledge and human effort in feature design is a major advantage.

They have applications in image and video recognition, recommender systems,image classification, medical image analysis, and natural language processing.(source: <a href="https://en.wikipedia.org/wiki/Convolutional_neural_network"> wikipedia </a>) 
 
 **Example:**
 
   Given a set of dog and cat pictures , train a CNN to predict in which of these two categories a new picture belongs to.
   The code can be found <a href="https://github.com/gpsyrou/Deep-Learning-A-Z/blob/master/Convolutional%20Neural%20Networks/classify_catsndogs.py"> here </a>.
   
   **Readings**
   <ul>
      <li>Yann LeCun et al., 1998,&nbsp;<a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"><em>Gradient-Based Learning Applied to Document Recognition</em></a></li>
      <li>Jianxin Wu, 2017,&nbsp;<i><a href="http://cs.nju.edu.cn/wujx/paper/CNN.pdf">Introduction to Convolutional Neural Networks</a></i></li>
      <li>C.-C. Jay Kuo, 2016,&nbsp;<i><a href="https://arxiv.org/pdf/1609.04112.pdf">Understanding Convolutional Neural Networks with A Mathematical Model</a></i></li>
      <li>Kaiming He et al., 2015,&nbsp;<i><a href="https://arxiv.org/pdf/1502.01852.pdf">Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</a></i></li>
      <li>Dominik Scherer et al., 2010,&nbsp;<i><a href="http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf">Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition</a></i></li>
      <li>Adit Deshpande, 2016,&nbsp;<i><a href="https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html">The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)</a></i></li>
      <li>Rob DiPietro, 2016,&nbsp;<i><a href="https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/">A Friendly Introduction to Cross-Entropy Loss</a></i></li>
      <li>Peter Roelants, 2016,&nbsp;<a href="http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/"><i>How to implement a neural network Intermezzo 2</i></a></li>
   </ul>


<br></br>
**SECTION 3 - Recurrent Neural Networks (RNN)**

<img src="https://cdn-images-1.medium.com/max/1600/1*6xj691fPWf3S-mWUCbxSJg.jpeg" alt="22" style="width:104px;height:142px;">
  

A recurrent neural network is a class of artificial neural network where connections between nodes form a directed graph along a sequence. This allows it to exhibit temporal dynamic behavior for a time sequence. Unlike feedforward neural networks, RNNs can use their internal state to process sequences of inputs. (source: <a href="https://en.wikipedia.org/wiki/Recurrent_neural_network"> wikipedia </a>)

**Example:**

Predicting the trend of Google's stock price using LSTM.
The code for this excercise can be found <a href="https://github.com/gpsyrou/Deep-Learning-A-Z/blob/master/Recurrent%20Neural%20Networks/recurrent_nn.py"> here </a>.

<br></br>
**SECTION 4 - Self Organizing Maps (SOM)**
