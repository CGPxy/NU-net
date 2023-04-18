# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from model_net import *
matplotlib.use("Agg")
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


filepath = args['filepath']

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
    else:
        # img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
        # print(img.shape)
    return img

def get_train_data():
    train_url = []
    train_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    print(total_num)
    for i in range(len(train_url)):
        train_set.append(train_url[i])

    return train_set


def get_train_val(val_rate=0.25):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set

# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, valid_label)
                valid_data = []
                valid_label = []
                batch = 0  

class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor,
                                                      verbose, save_best_only, save_weights_only, mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)



from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
  """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
  if hasattr(model, 'inputs'):
    try:
      # Get input shape and set batch size to 1.
      if model.inputs:
        inputs = [
            tf.TensorSpec([1] + input.shape[1:], input.dtype)
            for input in model.inputs
        ]
        concrete_func = tf.function(model).get_concrete_function(inputs)
      # If model.inputs is invalid, try to use the input to get concrete
      # function for model.call (subclass model).
      else:
        concrete_func = tf.function(model.call).get_concrete_function(
            **inputs_kwargs)
      frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

      # Calculate FLOPs.
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      if output_path is not None:
        opts['output'] = f'file:outfile={output_path}'
      else:
        opts['output'] = 'none'
      flops = tf.compat.v1.profiler.profile(
          graph=frozen_func.graph, run_meta=run_meta, options=opts)
      return flops.total_float_ops
    except Exception as e:  # pylint: disable=broad-except
      logging.info(
          'Failed to count model FLOPs with error %s, because the build() '
          'methods in keras layers were not called. This is probably because '
          'the model was not feed any input, e.g., the max train step already '
          'reached before this run.', e)
      return None
  return None


def train(args):
    EPOCHS = 50
    BS = 10

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        model = deep_unet7()
        model.summary()
        flops = try_count_flops(model)
        print(flops/1000000000,"GFlops")

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) # 二分类binary_crossentropy

    checkpointer = ModelCheckpoint(os.path.join(
        args['save_dir'], 'model_{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=False, mode='max')

    tensorboard = TensorBoard(log_dir='./logs/deep_unet7/BUSI/1/', histogram_freq=0, write_graph=True, write_images=True)

    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=[checkpointer, tensorboard])

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--filepath", required=True, help="path of your train data")
    ap.add_argument("--save_dir", required=True, help="path to output model")
    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = args_parse()
    train(args)
