# TensorFlow Object Detection API Tutorial for Traffic Light Detection(TLD)







1. Set your environment

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

```
git clone https://github.com/tensorflow/models.git 
```

```
# From tensorflow/models
vi research/object_detection/g3doc/installation.md
```

```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
```

```
# For CPU
pip install tensorflow
```

```
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

```
python object_detection/builders/model_builder_test.py
```

2. Test a Model

- Save images to models/research/object_detection

- Open models/research/objection_detection/**object_detection_tutorial.ipynb**

- Edit TEST_IMAGE_PATHS

```
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, N+1) ]
```

- Here is my result: [A result of the jupyter notebook](https://github.com/OliverPark/Traffic-Light-Detection/blob/master/research/object_detection/object_detection_tutorial.html)


3. Prepare train images

- [Using_your_own_dataset.md](https://github.com/tensorflow/models/blob/f20630e974b9ef0d44f45067360c7180db455f22/research/object_detection/g3doc/using_your_own_dataset.md)

- Sample .yaml

```
- annotations:
  - {class: Green, x_width: 52.65248226950354, xmin: 130.4964539007092, y_height: 119.60283687943263,
    ymin: 289.36170212765956}
  - {class: Green, x_width: 50.156028368794296, xmin: 375.60283687943263, y_height: 121.87234042553195,
    ymin: 293.90070921985813}
  - {class: Green, x_width: 53.33333333333326, xmin: 623.6595744680851, y_height: 119.82978723404256,
    ymin: 297.7588652482269}
  class: image
  filename: sim_data_capture/left0003.jpg
```

- Make TFRecord files


```
# From models/research/object_detection
python tf_record_sim.py --output_path test_images/training_sim.record
python tf_record_real.py --output_path test_images/training_real.record
```

4. Training the Model

- Mapping Data: models/research/objection_detection/**traffic_light_map.pbtxt**

```
  item {
    id: 0
    name: 'Red'
  }

  item {
    id: 1
    name: 'Yellow'
  }

  item {
    id: 2
    name: 'Green'
  }

  item {
    id: 4
    name: 'Undefined'
  }
```




















References: https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e

