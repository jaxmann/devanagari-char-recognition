<!DOCTYPE html>
<html lang="en"><head>  
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="">
  <meta name="author" content="">

<!-- Le styles -->  
  <link href="css/bootstrap.css" rel="stylesheet">

  
<link href="css/bootstrap-responsive.min.css" rel="stylesheet">
</head>

<body>
<div class="container">
<div class="page-header">

<!-- Title and Name --> 
<h1>Devanagari Character Recognition</h1> 
<span style="font-size: 20px; line-height: 1.5em;"><strong>Jonathan Axmann, Chris Collins, Mihir Shrirang Joshi</strong></span><br>
<span style="font-size: 18px; line-height: 1.5em;">Fall 2018 CS 7643 Deep Learning: Class Project</span><br>
<span style="font-size: 18px; line-height: 1.5em;">Georgia Tech</span>
<hr>

<!-- Goal -->
<h2>Abstract</h2>

We attempt to perform image classification on the Devanagari character set. For an input of handwritten Devanagari symbols, our model will output the category label (character name) of each character. We will use a variety of models, including a small fully-connected net with two layers, a simple convolutional net with one convlutional layer, and more complex models, as well as pretrained-models, and compare performance. The goal of this project is to achieve a similar accuracy to the papers outlined below while using simpler models and fewer training epochs. 

<br><br>
<!-- Introduction -->
<h2>Introduction</h2>
<h4>What did you try to do? What problem did you try to solve?</h4>
Our project aimed to solve the problem of classifying Devanagari script. 
<br><br>
Devanagari is an Indic script used in India and Nepal, and our dataset contains 36 characters and 10 digits. It is differentiable from many other written languages by the lack of capitalization and the horizontal bar aligned along the top of the script. Below are some examples of input used for this project. These are handwritten symbols from the training dataset used to train our model. Note that each of the following are different characters, although some of them appear quite similar - this is the problem that our model will attempt to resolve. Since these are handwritten characters, some amount of error can be attributed to sloppy or perhaps illegible writing - it might not be possible for even a human subject-matter expect to achieve 100% accuracy. 
<br><br>

<table border=0 style="width:100%">
<tr>
<td>
<img src="project_webpage/images/examples/1340.png" width="4.8%"/>
<img src="project_webpage/images/examples/2771.png" width="4.8%"/>
<img src="project_webpage/images/examples/3710.png" width="4.8%"/>
<img src="project_webpage/images/examples/3805.png" width="4.8%"/>
<img src="project_webpage/images/examples/3946.png" width="4.8%"/>
<img src="project_webpage/images/examples/4079.png" width="4.8%"/>
<img src="project_webpage/images/examples/117.png" width="4.8%"/>
<img src="project_webpage/images/examples/192.png" width="4.8%"/>
<img src="project_webpage/images/examples/338.png" width="4.8%"/>
<img src="project_webpage/images/examples/469.png" width="4.8%"/>
<img src="project_webpage/images/examples/608.png" width="4.8%"/>
<img src="project_webpage/images/examples/720.png" width="4.8%"/>
<img src="project_webpage/images/examples/804.png" width="4.8%"/>
<img src="project_webpage/images/examples/964.png" width="4.8%"/>
<img src="project_webpage/images/examples/1083.png" width="4.8%"/>
<img src="project_webpage/images/examples/4200.png" width="4.8%"/>
<img src="project_webpage/images/examples/4317.png" width="4.8%"/>
<img src="project_webpage/images/examples/4434.png" width="4.8%"/>
</td>
</tr>
</table>



<br>
Our training dataset contains 2000 examples of each character, for a total of 92,000 images. Each image consists of 32x32 pixels, and 3 color channels. The test set consists of 13800 total images (300 for each character) and the training set consists of 78200 images (1700 per character). This accounts for an 85/15 split. 


<h4>How is it done today, and what are the limits of current practice?</h4>
Current attempts to classify Devanagari script are outlined here:


 - [https://towardsdatascience.com/devanagari-script-character-recognition-using-machine-learning-6006b40fa6a9](https://towardsdatascience.com/devanagari-script-character-recognition-using-machine-learning-6006b40fa6a9) (by Rishi Anand)
This project uses an Extremely Randomized Decision Forest (but not deep learning) to classify handwritten Devanagari script. It achieves a maximum accuracy score of 92%. 

 - [https://www.hindawi.com/journals/cin/2018/6747098/](https://www.hindawi.com/journals/cin/2018/6747098/) (by Alom et al.)
This project compares accuracy of VGG (97.6%), ResNet (97.3%), DenseNet (98.3%), and FractalNet (97.9%) over 250 epochs and across a variety of output groupings (i.e. digit-only, vowel-only, and all-character). 

We (and both of the aforementioned projects) acquired data from:  [https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset)


<h4>Who cares? If you are successful, what difference will it make?</h4>
Character recognition (for any language) is important to transcribe written text into digital representations. For many difficult-to-read handwritten instances of lettering, human intervention is required via methods such as Captcha, Gamification, or manual annotation. This process is often labor and cost-expensive. However, for many character sets, deep learning models can accurately transcribe handwritting to digital encoding. If our project is successful, it could be useful in digitizing literature in Devanagari script, or as a template to do so in a lightweight fashion, without requiring extensive GPU compute time to train models. 

<br><br>
<!-- Approach -->
<h2>Approach</h2>
<h4>What did you do exactly? How did you solve the problem? Why did you think it would be successful? Is anything new in your approach?</h4>
The goal of our project is to achieve comparable accuracy to the models used in the Alom et al. paper above, but using a simpler model and fewer training epochs. Ideally, simply tuning hyperparameters and intelligent design decisions will allow us to achieve high accuracy with a less complex model (or a pre-trained model) to reduce overall training time. Our approach could be useful to apply to character recognition tasks when there are limited resources to train with. The paper above trains each model on 250 epochs to achieve a maximal accuracy of approximately 98%. We will limit our maximum training epochs to 25, and see how competitive our accuracy will be. The best VGG-accuracy the paper achieves is 97.6%. 
<br><br>
We tried a variety of models, including a simple two-layer fully-connected net, a simple convolutional net, variants of VGG11 (using different input sizes and additional max-pooling layers), and several pre-trained AlexNets. Our approach was to first use simple models to make sure the data was formatted correctly and that our approach would work at all, as a sanity check, and then move on to more complex models. The novelty of our approach is that we will use significantly fewer training epochs compared to the standard number of 250 in the Alom et al. paper - we expect this will relatively successful given the marginal rate of returns for each successful epoch (i.e. the loss and accuracy curves level off after a certain number of training epochs). 


<h4>What problems did you anticipate? What problems did you encounter? Did the very first thing you tried work?</h4>
One problem that we faced was having to change the input sizes to some of the pre-trained networks. VGG11, for instance, takes a 224x224x3 image as input, and our images are 32x32x3. It's possible to work around this problem by simply resizing the images for the pre-trained models. For fully-trained VGG11, however, it is not feasible to use 224x224x3 images since the memory usage is extremely high. For this reason, we created a VGG11-variant that accepted 32x32x3 input images and still achieved high accuracy. 

<br><br>
<!-- Results -->
<h2>Experiments and Results</h2>
<h4>How did you measure success? What experiments were used? What were the results, both quantitative and qualitative? Did you succeed? Did you fail? Why?</h4>

We measured success by measuring how many of the test set images were correctly categorized into their respective category bin out of all the categories. Random accuracy would be about 2.2% accuracy (1 out of 46). There were 13800 total images in the test set and our best model correctly categorized 13556 of them with its first guess. We chose not to use top5 accuracy because it does not make sense to allow to model to guess multiple times on character recognition - it's important to be correct on the first try. Regardless, accuracy is already so high top5 accuracy is likely 100% or close to it, so we are using top1 accuracy. 
<br>
<br>
Our goal was to match the performance of the deep learning paper by Alom et al. that achieved 98% accuracy, but do so with a simpler model. We feel that we have achieved this goal. The  article by Rishi Anand listed above uses only Extremely Randomized Forests and achieves 92% accuracy, which is fairly easy to beat with pre-trained AlexNet (achieving 96% accuracy). The deep learning model in the Alom et al. paper above uses several models, such as VGG, ResNet, and DenseNet. These are all trained over at least 250 epochs. Our goal was to use at most a tenth the amount of epochs along with a simpler model to prove that in cases where compute resources are perhaps more limited, it is still feasible to train a deep net to perform a complex classification task. The key to our approach was to modify a variant of VGG11 to take smaller dimensional input rather than the default of 224x224x3 and instead use 32x32x3. This allows the model to train significantly faster (and use far less memory resources), while maintaining similar accuracy in this case. 

<br><br>

<!-- Main Results Figure --> 
<!-- <div style="text-align: center;"> -->

<h3>Two Layer FC (46% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/twolayernn_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/twolayernn_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<!-- <div  style="width:35%">
<img src="project_webpage/images/twolayernn32d.png" />
<p style="font-size: 14px">First-layer filter visualization for two layer FC</p>

 </div> -->
<br>
Initially, we achieved a baseline accuracy by using a simple model that combines two fully-connected layers (with a ReLu in-between). These act as both the feature extractor and classifier at once. We use this as a minimum threshold that other models should achieve (and outperform). 
<br><br>

<h3>FC Layer with Softmax Classifier (70% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/softmax_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/softmax_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<!-- <p style="font-size: 14px">Filter visualization for softmax classifier</p>
<div  style="width:35%">
<img src="project_webpage/images/softmax_filt.png" />
</div> -->
<br>
This model is simply a single linear layer that uses softmax as the classifier. 
<br>
<br>


<h3>Single Conv Layer with 224x224x3 Input (80% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/convnet224d_80_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/convnet224d_80_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<br>
The next-most complex model we created was a simple single-layer convolutional network. It performs a convolution and maxpool as the feature extraction segment, and uses a fully-connected layer as the classifier. This performs very well for how simple the model is. In this case, we re-sized the images to 224x224x3 using a PyTorch transformation. This was mostly a sanity check to see what effect re-sizing would have, given that we would need to do it for inputs to AlexNet. 
<br><br>

<h3>Single Conv Layer with 32x32x3 Input (88% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/convnet32d_88_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/convnet32d_88_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<div  style="width:35%">
<img src="project_webpage/images/convnet32d.png" />
<!-- <p style="font-size: 14px">Filter visualization for simple convolutional net with 32x32x3 input</p> -->

 </div>

 Using the default 32x32x3 images we used as data input, we achieved 88% accuracy using the extremely simple single-layer convolutional network. This nearly matches the Random Forest model used by the Rishi Anand article that applied a variety of optimizations. Our more complex networks should significantly outperform this accuracy threshold. The filters learned do not appear to be particularly meaningful to us.
 <br><br>


<h3>Fully-Trained VGG11 Variant with 32x32x3 Input (10 epochs) (97% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/vgg11_10ep_var_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/vgg11_10ep_var_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<div style="width:35%">
<img src="project_webpage/images/vgg11_10ep_var.png" />
<!-- <p style="font-size: 14px">Filter visualizations for variant of vgg11 with 32x32x3 input</p>
 -->
</div>

We experimented with both fully-trained and pre-trained networks. Since it was not feasible to train the normal VGG11 architecture from scratch on 224x224x3-resized images (the memory usage is enormous), we modified the network to accept 32x32x3 (but use mostly the same architecture, with an additional average pooling layer at the end, which was found empirically to work well). We also intentionally used a small number of epochs to train as the purpose of this project is to replicate the results from the Alom et al. paper using significantly less resouces. This achieved an accuracy score of 97%.  
<br><br>


<h3>Fully-Trained VGG11 Variant with 32x32x3 Input (15 epochs) (97.8% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/vgg11_15ep_var_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/vgg11_15ep_var_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<br>
As above, running this same model for 5 additional epochs produced the accuracy score of 97.8%. It correctly classified 13506 out of 13800 characters in the test set. This approaches the goal set by the Alom et al. paper of 98%, and is likely roughly what a human could achieve. The Alom et al. paper, however, only achieves 97.6% accuracy using the VGG model. In this case, it appears our model has performed better. 
<br><br>

<h3>Pre-Trained AlexNet (last layer tuned - 10 epochs) (97% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/alexnet_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/alexnet_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>

<br>
We also tried using pre-trained networks. AlexNet was the most convenient to use, although it required us to resize the image to 224x224x3. Fortunately, we are only replacing the final layer of the classification layers with a new layer that has the correct output dimensions (to match the number of classes, 46, in our case). In doing so, we forced the network to update weights (i.e. backprop) over the last 3 layers. 
<br><br>


<h3>Pre-Trained AlexNet (last two layers tuned - 10 epochs) (98.2% accuracy)</h3>

<table border=1>
<tr>
<td>
<img src="project_webpage/images/alex2_982_lossvstrain.png" width="49%"/>
<img src="project_webpage/images/alex2_982_valaccuracy.png"  width="49%"/>
</td>
</tr>
</table>


<br>
For our final model, we again used transfer learning with AlexNet. In this case, we completely re-initialized weights for the last two layers. We required the model to backprop weight updates over the last 5 classification layers. Again, note that we only used 10 epochs and managed to nearly equal the best accuracy of the Alom et al. of 98.3% by correctly predicting 13556 out of 13800 test examples for an accuracy score of 98.2%
<br><br>



</div>
<br><br>
  
  <hr>
  <footer> 
  <p>© Jonathan Axmann, Chris Collins, Mihir Shrirang Joshi</p>
  </footer>
</div>
<!-- </div> -->

<br><br>

</body></html>
