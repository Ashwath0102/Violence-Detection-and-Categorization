# Violence-Detection-and-Categorization

The purpose of violence detection is to determine whether or not a violent action has occurred. This topic grew in popularity due to the need to develop appropriate and automatic violence detection systems that investigated visual data obtained from security cameras positioned in various regions. In this work, we used pre-trained deep neural networks to provide a low-complexity strategy for detecting violence. To detect whether a violent action happened, the extracted features from pre-trained models were pooled and given into a fully-connected layer. Even with the most powerful and accurate methodologies, at the end of the day it all hinges on the fact that if the footage is captured flawlessly with perfect vision, the experimental results demonstrated the efficiency of the suggested low-complexity technique, sequential style methodology, is a trustable, sturdy, and efficient model for the job of sensing and categorizing four distinct violent events in comparison to existing approaches that use time-consuming systems such as recurrent networks.

## Problem Statement

Since the beginning of time, violence has been a part of this planet. It's been that manner ever since humanity's inception. Tim Larkin has said that bloodshed will mostly never be the way out, but one time or one day it's the only answer and ignoring it will lead to a bad situation.There has been tremendous development in technology across all fields. It has affected all walks of life, and we can see the fruits it has borne. This can be especially seen in surveillance systems. We're able to view and gather additional data than it has ever been. Despite this, crimes are still a common occurrence. We believe that there's still untapped potential in surveillance systems. We place a high value on CCTV and surveillance systems, which are found at airports, malls, train stations, homes, and other public spaces. As we all know, human error can occur, and a crime can occur in a fraction of a second, which may go unnoticed by a worker who is watching. Therefore, to increase the sense security among public we present a completely unique feature using transfer learning methodology and our CNN model that's supported by means of a Spatio-temporal model and uses appearance, background motion compensation, and long-term motion information. 

## Violence Detection architecture

![ArchitecfinalsemFINAL](https://user-images.githubusercontent.com/59199696/169958695-904278c7-4038-4b18-97de-f500df118fe1.jpg)
  
                        
The main objectives of our architecture are to handle pose estimation, recognition, and classify whether an action is violent or not, and if it's violent, we'll further attempt to classify it from the subsequent four classes that are: arrest, assault, arson (purposely setting properties on fire), and abuse. 
The final module is training, which has two parts. 
1.	The first implementation is our own way, which is our normal CNN model, which has 3 dense hidden layers with Relu as their activation functions, and every single layer is accompanied by dropouts to challenge the system. The final output is done through the SoftMax activation function for our multi-class problem in order to classify violence into the four classes: arrest, assault, arson, and abuse, and the one with the highest probability value is the violent action if it's greater than 0, that is, if it's violent, which returns a probability distribution which adds up to 1. The results were kind of overfitting with a 0.9992 ROC AUC score. 

2.	The second method employs a pre-trained model. The system's structure is made up of CNN, which is made up of prediction blocks (PB), downscaling units (DU), and upscaling units (UP) (UU). The Prediction block (PB) within the CNN handles the pose prediction and action prediction. Once the system training is done, the transfer learning procedure is used. The key advantage of the transfer learning methodology is that it scales better for the current application of violence recognition and has low computing demand and complexity. We scaled the frames before delivering them to the model to acquire the features. With the help of a feature vector, we perform max pooling, which is usually added at the end of a CNN. This is done to further decrease the computational load and once again improve the overall efficiency. The resultant output is a feature vector after the transfer learning method is applied. Finally, the output is fed into a convolution neural network with ReLu and, as before, it has a SoftMax output layer, which returns a probability distribution which adds up to 1, as mentioned in the first part. As we know, the final goal is to classify whether or not an action is violent. Hence, if the final output corresponds to a probability distribution from the values 0 to 1, which adds up to 1, as the figure below, the action is classified as violent and with the highest probability class as the output from those four violent classes, whereas if the final output corresponds to Zero (0), the ferocious lawbreaking action is classified as non-violent. 

![image](https://user-images.githubusercontent.com/59199696/169957662-6ef398a9-28fc-4fc7-94c3-3d9603ba451f.png)



### Dataset Description

To classify the videos into four different crime categories or non-violent, we downloaded 200 videos from various sources such as news channel recordings, YouTube, and online posted videos on all other social networking sites, and created four different classes, such as arrest, assault, arson, and abuse, with each class containing 50 videos. We then converted each footage into its frames, which were subsequently turned into images with a width and height of 64 and 64, respectively. In the appropriate crime categories, we received 63,060 images for arrest, 16,177 images for assault, 126,553 images for arson, and 28,476 images for abuse. 

##### Train Data Distribution
![image](https://user-images.githubusercontent.com/59199696/169960724-b831287f-f353-4e7d-80d4-f614f43c4dfc.png)

##### Test Data Distribution
![image](https://user-images.githubusercontent.com/59199696/169960827-d97167ad-a852-4d68-9d7b-ef4f3aaa9298.png)

We're using 64 batches, epochs of 1, and a learning rate of 0.0003. As shown in the above Figures, in total, we employed 187,414 images for the train set in which we received 50,448 images for arrest, 12,942 images for assault, 101,243 images for arson, and 22,781 images for abuse, and 46,852 images for the validation set in total for the data split in which we received 12,612 images for arrest, 3,235 images for assault, 2,531 images for arson, and 5,695 images for abuse. 

###### Dataset Link: https://www.kaggle.com/datasets/ashwathbaskar/violence-classification-for-the-4-crime-categories

### Hardware & Software Requirements

   Hardware Requirements:
   
   MINIMUM:
     
     OS: Windows 7
     Processor: Intel Core i3 Dual core | AMD Phenom II X4 965
     Memory (RAM): 8 GB 
     Graphics: Nvidia GeForce GTX 650, 1 GB | AMD Radeon HD
     HD 6950, 2 GB
     DirectX: Version 11
     Hard Drive: Minimum 100 GB
     Network: Ethernet connection (LAN) OR a wireless adapter (Wi-Fi)
     Storage: 8 GB available space


   RECOMMENDED:
    
    OS: Windows 10
    Processor: Intel Core i5-2300 | AMD FX-6300
    Memory (RAM): 32 GB or more
    Graphics: Nvidia GeForce GTX 660, 2 GB AMD Radeon HD
    7970, 3 GB
    DirectX: Version 11
    Hard Drive: 200 GB or more
    Network: High-speed Ethernet connection (LAN) OR a wireless adapter (Wi-Fi)
    Storage: 12 GB available space.



   Software Requirements:
    
    Python
    Anaconda
    Jupyter Notebook
    Pip 19.0 or later
    CUDA®-enabled card
    Browser: Firefox, Edge, chrome
    Sklearn, Plotly, Seaborn, Pandas, Numpy, Matplotlib, Datetime, Av, Cv2, Time, Os, TensorFlow, and Keras are the libraries we'll be utilising.


## Conclusion and Future Work

This solution demonstrates that using transfer learning is the optimal strategy for developing a reliable, stable, and efficient model for the job of detecting violence with such a small dataset and computational capabilities. The transfer learning + CNN sequential style methodology achieved a very good accuracy of 0.9364 for the violent categorization task, showing that this model is the finest solution to our paper. The journey, however, doesn't conclude here. We will use these models on various devices, like CCTV and unmanned aerial vehicles (UAV). With this framework and also the growth of technology, the prices of the equipment resources needed have become more affordable, and with the cooperation of presidency agencies, the system may well be further improved and made far more efficient by progressing with the plan of pruning the models so as to form them ready to be deployed on devices with low internal memory units, and that we could install violence sensors in schools, restaurants, airports, and other public places so the authorities will receive an alert on the server and display the live footage to them.


## References

[1]	D. C. Luvizon, D. Picard and H. Tabia, "Multi-Task Deep Learning for Real-Time 3D Human Pose Estimation and Action Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 8, pp. 2752-2764, 1 Aug. 2021, doi: 10.1109/TPAMI.2020.2976014.

[2]	Sárándi, István & Linder, Timm & Arras, Kai & Leibe, Bastian. (2020). MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation. arXiv:2007.07227.

[3]	A. Mumtaz, A. B. Sargano and Z. Habib, "Violence Detection in Surveillance Videos with Deep Network Using Transfer Learning," 2018 2nd European Conference on Electrical Engineering and Computer Science (EECS), 2018, pp. 558-563, doi: 10.1109/EECS.2018.00109.

[4]	Keze Wang, Liang Lin, Chenhan Jiang, Chen Qian, Pengxu Wei, "3D Human Pose Machines with Self-supervised Learning", jan 2019,	arXiv:1901.03798

[5]	Yiming Wang, Lingchao Guo, Zhaoming Lu, Xiangming Wen, Shuang Zhou, Wanyu Meng, "From Point to Space: 3D Moving Human Pose Estimation Using Commodity WiFi", Dec 2020,  	arXiv:2012.14066

[6]	Zhang, Zihao & Hu, Lei & Deng, Xiaoming & Xia, Shihong. (2020). Weakly Supervised Adversarial Learning for 3D Human Pose Estimation from Point Clouds. IEEE Transactions on Visualization and Computer Graphics. PP. 1-1. 10.1109/TVCG.2020.2973076.

[7]	H. Xia and M. Xiao, "3D Human Pose Estimation With Generative Adversarial Networks," in IEEE Access, vol. 8, pp. 206198-206206, 2020, doi: 10.1109/ACCESS.2020.3037829.

[8]	Guoqiang Wei, Cuiling Lan, Wenjun Zeng, Zhibo Chen, "View Invariant 3D Human Pose Estimation", Jan 2019, arXiv:1901.10841

[9]	R. Gu, G. Wang, Z. Jiang and J. -N. Hwang, "Multi-Person Hierarchical 3D Pose Estimation in Natural Videos," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 11, pp. 4245-4257, Nov. 2020, doi: 10.1109/TCSVT.2019.2953678.

[10] C. Yang, X. Wang and S. Mao, "RFID-Pose: Vision-Aided Three-Dimensional Human Pose Estimation With Radio-Frequency Identification," in IEEE Transactions on Reliability, vol. 70, no. 3, pp. 1218-1231, Sept. 2021, doi: 10.1109/TR.2020.3030952.

[11] G. Cheron, I. Laptev, and C. Schmid, "P-CNN: Pose-based CNN Features for Action Recognition," in ICCV, 2015.

[12] U. Iqbal, M. Garbade, and J. Gall, "Pose for action - action for pose," FG-2017, 2017. I. Kokkinos, "Ubernet: Training a universal convolutional neural net- work for low-, mid-, and high-level vision using diverse datasets and limited memory," Computer Vision and Pattern Recognition (CVPR), 2017.

[13] S. Park, S.-b. Lee, and J. Park, "Data augmentation method for improv- ing the accuracy of human pose estimation with cropped images," Pattern Recognit. Lett., vol. 136, pp. 244250, Aug. 2020.

[14] V. Choutas, P. Weinzaepfel, J. Revaud, and C. Schmid, "Potion: Pose motion representation for action recognition," in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

