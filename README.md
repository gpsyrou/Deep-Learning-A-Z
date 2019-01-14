# Deep-Learning-A-Z

<img src="http://uc-r.github.io/public/images/analytics/deep_learning/deep_nn.png" alt="22" style="width:104px;height:142px;">

## Notes from the course Deep Learning A-Z provided by Udemy.

**SECTION 1 - Artificial Neural Networks(ANN)**:

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

**SECTION 2 - Convolutional Neural Networks(CNN)**
   
 <img src="https://cdn-images-1.medium.com/max/1600/1*NQQiyYqJJj4PSYAeWvxutg.png" alt="22" style="width:104px;height:142px;">
  
 
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

**SECTION 3 - Recurrent Neural Networks(RNN)**

A recurrent neural network is a class of artificial neural network where connections between nodes form a directed graph along a sequence. This allows it to exhibit temporal dynamic behavior for a time sequence. Unlike feedforward neural networks, RNNs can use their internal state to process sequences of inputs. (source: <a href="https://en.wikipedia.org/wiki/Recurrent_neural_network"> wikipedia </a> )





