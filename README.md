# License Plate Recognition using CNN

## Contributors:

⚜️ [Shubham Bhalala](https://www.linkedin.com/in/shubhambhalala/) ⚜️ [Tirth Patel](https://www.linkedin.com/in/tirupatel/) ⚜️ [Harsh Modi](https://www.linkedin.com/in/harshkumar-modi-50a676158/) ⚜️ [Shashwat Misra](https://www.linkedin.com/in/shashwat-misra-07802814b/) ⚜️ [Shashwat Gaur](https://www.linkedin.com/in/shashwat-gaur-a0060516b/)

To achieve this task, let's understand the architecture. We need to recognise the license plate, here specifically Indian, all the contries have different type of license plate character and shape of plate.
So, overall we need to detect the number plate first, then recognise the characters. We also have one more bifurcation in this i.e. all the vehicle have different shape and size of number plate.
In this task, we are focusing only for car, and any number plate which is of the shape and size of car's number plate in India. Once, we have the plate recognised we have to recognise the characters of the plate.
Finally, we have to feed these recognised numbers to a API which can give us the vehicle information.

## WorkFlow:

### Model Creation:

Image --> HaarCascade [indian_license_plate](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/indian_license_plate.xml) --> Extracted Plate --> Character Segmentation --> [DataSet of characters of Indian License Plate](https://github.com/AnonMrNone/indian_licenseplate_recognition/tree/master/data/data) --> CNN Model Training --> model.predict(character segmented) --> Extracted plate number

Here, HaarCascade will only detect the number plate of size and shape related to car's number plate.

### Model Testing:

Import model and necessary function --> New Image of Car --> HaarCascade --> Extracted Plate --> Character Segmentation --> model.predict(segmented character) --> API

### Final webapp for video stream

Import model and necessary function --> Take Video From user --> For each frame --> HaarCascade --> Extracted Plate --> Character Segmentation --> model.predict(segmented character) --> API --> Final Output on Webpage

### API used for getting vehicle information

http://www.carregistrationapi.in/

## How to train the model?

As discussed earlier we have to train the model to recognise the plate's characters i.e. alphabets and numbers. Creating such a model need few things which will be explained in here.
The standard procedure to extarct any objects from the image is by finding the contours.

### What are contours?

Contours can be simply explained as a curve joining all the continuous points (along the boundary), having same colour or intensity. The contours are a useful tool for shape analysis and object detection and recognition.
* For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection. Since OpenCV 3.2 and later, findContours() no longer modifies the image source.
* In OpenCV, finding contours is like finding white object from black background. So remember, object to be found should be white and background should be black.

Read more at: https://docs.opencv.org/4.5.2/d4/d73/tutorial_py_contours_begin.html

![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/What%20are%20contours.png)

### Segment Characters:

Now, we know that we have to first convert image into binary image and then we have to perform the thresholding on it. ONce, we do that we even have to make our charectes grow perfect in the black background.
To do so, we have threshold function in cv2 module.

#### What is threshold?

For, every pixel, the same threshold value is applied. If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value. The function cv.threshold is used to apply the thresholding. The first argument is the source image, which should be a grayscale, second argument is the threshold value which is used to classify the pixel values, third argument is the maximum value which is assigned to pixel values exceeding the threshold. OpenCV provides different types of thresholding which is given in the fourth argument.
In our case we have used BINARY and OTSU together. BINARY will create an perfect contrast of black and white on the image and OTSU will remove all the noise from the image.

Read more at: https://docs.opencv.org/4.5.2/d7/d4d/tutorial_py_thresholding.html

![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/Binary.png)
![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/otsu.png)

#### What is erode and dilate?

We have already performed the threshold of the image, now to set all the characters come brightly in the image and there are perfect edges and curves, we need to perform morphological operations.
These two morphological operations will help us make the characters become more clear. Eg: D won't appear as O 

Erode and dilate are most basic morphological operations, morphological operations are a set of operations that process image based in shapes, morphological operations apply a structuring element to an input image and generate an output image. Erode and dilate have wide range of uses:
* Removing noise
* Isolation of individual elements and joining disparate elements in an image.
* Finding of intensity bumps or holes in an image.

##### Dilation:

This operations consists of convolving an image A with some kernel B, which can have any shape or size, usually a square or circle.
Doing this operation the brighter portion of the image grows.

![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/dilate.png)

#### Erode:

This is exactly opposite of Dilation, it computes a local minimum over the area of given kernel.

![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/erode.png)

Read more at: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

### What is F1 Score?

It’s just a metrics using which we can measure our model’s accuracy. So, what’s problem with accuracy metrics? Well the answer is very simple, let’s say in our data set, values/records are such that, it will favour positive side more rather than negative. In layman’s term, we might have 80% data which is in favour of our prediction and only 20% which is against, so obviously our model will be biased to one. In these cases, we might get more false positive as well as false negative. Overall we need to avoid Type 1 error more than Type 2. To do so, we have F1 Score. F1 Score is the harmonic mean of precision and recall.

![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/precision_recall.png)
![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/f1score.png)
![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/f1beta.png)

Read more at: https://towardsdatascience.com/f-beta-score-in-keras-part-i-86ad190a252f

### What is Dropout layer?

The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
Note that the Dropout layer only applies when training is set to True such that no values are dropped during inference. When using model.fit, training will be appropriately set to True automatically, and in other contexts, you can set the kwarg explicitly to True when calling the layer.
In simple words, The Dropout layer is a mask that nullifies the contribution of some neurons towards the next layer and leaves unmodified all others. We can apply a Dropout layer to the input vector, in which case it nullifies some of its features; but we can also apply it to a hidden layer, in which case it nullifies some hidden neurons.
Dropout layers are important in training CNNs because they prevent overfitting on the training data. If they aren’t present, the first batch of training samples influences the learning in a disproportionately high manner. This, in turn, would prevent the learning of features that appear only in later samples or batches.

![alt text](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/readme_images/droupout.png)

## Code according to the steps:

Model Training: [code](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/license_recognition.ipynb)

Model Testing: [code](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/testing_of_model.ipynb)

Final video stream test: [code](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/testing_of_model_video_live_stream.ipynb)

Final WebApp: [code](https://github.com/AnonMrNone/indian_licenseplate_recognition/tree/master/license-webapp)

## Demo Video:

[Video Link](https://github.com/AnonMrNone/indian_licenseplate_recognition/blob/master/license-webapp/Demo-Video.mp4)
