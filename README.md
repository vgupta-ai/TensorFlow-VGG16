# TensorFlow-VGG16
This repository loads the Vgg16 model and does prediction over it. 

The code has been mainly taken from <a href="https://github.com/machrisaa/tensorflow-vgg">tensorflow-vgg</a> and modified as per the requirement.

The converted <a href="https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM">VGG16NPY</a> have to be downloaded and saved in "model" folder with name "vgg16.npy".

The test images need to be placed in the "images" folder. The following command does the inference:

```python
python Vgg16Test.py 
```

