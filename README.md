This project uses [ORB](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html) in OpenCV-python.

## How to use:
Usage is very simple and straight forward.
1. Add training images to Imtrain.
2. Add test images to Imtest. The images in Imtest will be related to the images in Imtrain.
3. Run main.py
4. Here is an example output of the code

   [![Matching images][1]][1]


  [1]: https://i.stack.imgur.com/295Kg.png

## How it works
1. The training image is processed in order to look soft and have detectable corners.
2. ORB is used to get the features (here, 50000)
3. Good features are extracted using BFmatcher. The metric is the distance calculated in the ORB algorithm.
4. Finally, the image in the test set get their good features with each image in Imtrain and the image with the most good features wins and is displayed.
