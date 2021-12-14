# DoodleAI - AI based doodle classifier

**Live App:** https://doodleai.herokuapp.com

**🔴IMPORTANT❗** Under free tier, Heroku app goes to sleep after 30 min of inactivity, hence the website might be slow to load initially. <span style="color = 'yellow'">**Also make sure your graphic drivers are up to date and your browsers are enabled to use hardware acceleration. While testing, I found that the model does not work well on mobile browsers and the ones on desktop whose hardware acceleration feature was disabled.**</span>

## Demo
![ezgif com-gif-maker](https://user-images.githubusercontent.com/44807945/104150519-14c32a80-5400-11eb-99f9-949163feda34.gif)

I am sure, you can draw better doodles :wink:.
## Description

This is a classifier to predict label of hand drawn doodle images in real time. The idea is based on [QuickDraw](https://quickdraw.withgoogle.com/#) by Google. The [dataset](https://github.com/googlecreativelab/quickdraw-dataset) they provide contains 50 million images across 345 categories! I am using a subset of 50 categories for my model because of limited resources but one can use the same code with small tweaks to train on all 345 categories.

I built and trained the model in Pytorch and converted it to onnx format to use it in the browser. Initially my plan was to perform the classification on the backend. After drawing, the user would press a button and the request would be sent to the server for classification. That is how I built the app. However, it was very expensive on the server because for every image there would a request. Hence I decided to move the classification on the frontend. Also it is a lot more fun to see the model try to classify the image in real time :grin:.

It took me a very long time to train and tweak the model to obtain a good accuracy particularly because the dataset was huge even though I was using only a subset of it and training the model on the GPU. How someone draws a certain object varies a lot. It is all based on imagination and perception of that person about that object. Hence it was necessary to use lots of images per category to capture maximum variations. 

For the record, 88% was the average test accuracy(averaged out across the classes). Had it been allowed to train for longer, I believe it could have crossed 90% mark but I had already spent days on training it so I let it go. Yes days! Colab has a limit on the usage of GPUs after which the runtime gets disconnected. So I had to train the model for some hours every day for about 2-3 days to get good accuracy. Then I would make tweaks to the model and restart the process. 


## Model Architecture
![Architecture](https://user-images.githubusercontent.com/44807945/103314029-cf8a1a80-4a47-11eb-9210-0040b1d7af80.png)

I used a deep network of convolutional layers, hence to prevent overfitting I am using dropout layers after every pooling layer. This increases sparsity of the model. Towards the end, I am using average pooling layer to reduce the dimensions of image to 1x1 with 256 channels. This is based on concept of [Global Average Pooling](https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/) as seen in recent state-of-the-art CNN models such as InceptionNet and ResNet. Finally a fully connected layer use these 256 feature input to give 50 output scores. The detailed description and code is provided in the jupyter notebook.

## References and resources

 - https://github.com/googlecreativelab/quickdraw-dataset
 - https://github.com/yining1023/doodleNet
 - https://ml4a.github.io/guides/DoodleClassifier/
 - https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
 - https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks
 - https://alexisbcook.github.io/2017/global-average-pooling-layers-for-object-localization/
 - https://arxiv.org/pdf/1312.4400.pdf Section 3.2 and 4.6 on Global Average Pooling

