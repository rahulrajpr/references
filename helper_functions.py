### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.

import tensorflow as tf

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  
  plt.xticks(rotation = 90, fontsize = text_size)
  plt.yticks(fontsize = text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
  
# Make a function to predict on images and plot them (works with multi-class)
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
  
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Plot the validation and training data separately
import matplotlib.pyplot as plt

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
  
# Create function to unzip a zipfile or untar a .tgz file into current working directory 
# (since we're going to be downloading and unzipping a few files)

import zipfile
import tarfile
import os

def unzip_untar_data2(filename, dest_folder=None, remove_source_file_after_extraction=True):
    """
    Unzips (.zip file) or untar (.tgz or .tar.gz) file into the specified destination folder.
    Args:
        filename (str): a filepath to the target zip or tar folder to be extracted.
        dest_folder (str): destination folder path where the contents will be extracted. Default is None.
        remove_source_file_after_extraction (bool): whether to remove the source file after extraction. Default is True.
    """
    if dest_folder is None:
      dest_folder = os.path.splitext(filename)[0]  # Use the name of the source file without extension as the destination folder

    if filename.endswith('.zip'):
      print('File extension: .zip')
      print('File extracting.......')
      with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
        print('\nExtraction Completed successfully')

    elif filename.endswith('.tgz') or filename.endswith('.tar.gz'):
      print('File extension: .tgz or .tar.gz')
      print('File extracting.......')
      with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(dest_folder)
        print('\nExtraction Completed successfully')

    else:
      print('Error: Unexpected file extension ---> extension not listed in [.zip, .tgz, .tar.gz]')
      print('Hint: Task cannot be completed, check the file and file extension')

    if remove_source_file_after_extraction:
      os.remove(filename)
      print('Deleted source file')
    else:
      print('Source file reatined')

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to Retun the performance metrics of classification/ regression models

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred,is_regression = False):
  
  """
  Returns the dictionary performance metrics of classications and regression models

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array
      is_regression : default False returns classication metrics, if True, returns regression metrics

  Returns a dictionary of accuracy, precision, recall, f1-score.
  
  """
  if not(is_regression):
    # Classification metrics block
    model_accuracy = accuracy_score(y_true, y_pred)
    # Calculate model precision, recall and f1 score using "weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                    "precision": model_precision,
                    "recall": model_recall,
                    "f1": model_f1}
  else:
    # Regression metrics block
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)

    if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
      mae = tf.reduce_mean(mae)
      mse = tf.reduce_mean(mse)
      rmse = tf.reduce_mean(rmse)
      mape = tf.reduce_mean(mape)

    model_results =  {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy()}
  
  return model_results
  
# function for checking the random image and the corresponding augmented one

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import numpy as np
import tensorflow as tf

def augmented_image_random_view(train_dir,augmentation_model,img_size = (224,224), color_channel = 3):
  
  """display the random image and its augmented version returns from another augemtation model"""

  classes = os.listdir(train_dir)
  classes = [x for x in classes if x.find('.') < 0]
  num_classes = len(classes)
  random_class_id = random.choice(range(0,num_classes))
  random_class = classes[random_class_id]

  random_class_path = train_dir+'/'+random_class
  images = os.listdir(random_class_path)
  num_images = len(images)
  random_image_id = random.choice(range(0,num_images))
  random_image = images[random_image_id]

  random_image_path = random_class_path+'/'+random_image
  
  print('picks image randomly from random classes of the train directory')

  plt.figure(figsize = (14,5))
  plt.subplot(1,2,1)
  image_value = mpimg.imread(random_image_path)
  plt.imshow(image_value)
  plt.axis(False)
  image_shape = image_value.shape
  plt.title('normal image'+str(image_shape))

  plt.subplot(1,2,2)

  if np.max(image_value) > 1:
    image_value = image_value/255
  else:
    image_value = image_value

  tensor_resized = tf.image.resize(image_value,size = img_size)
  tensor_reshaped = tf.expand_dims(tensor_resized, axis = 0)
  tensor_augmented = augmentation_model(tensor_reshaped)
  
  image_augmented = tf.squeeze(tensor_augmented)
  plt.imshow(image_augmented)
  image_augmented_shape = image_augmented.shape
  plt.title('augmented image'+str(image_augmented_shape))
  plt.axis(False)

  
def plot_time_series(timesteps, values, title = None, label = None, figsize = None, fontsize = 14, offset=0):
  
  """
  Objective 
  ---------
  Function to plot the timeseries data in a scatterplot
  
  Parameters 
  ----------
  Timesteps : The time varient in the dataset
  values : values to be plotted against the time varient

  Note 
  ---------
  incase figsize or title values are provided, keranal retuns a new figure, 
  deafult value is None, which does not return a new figure, but gets plotted into the existing figure if already existing
  
  """
  x = pd.Series(timesteps).iloc[offset:]
  y = pd.Series(values).iloc[offset:]
  
  if figsize == None:
    plt.plot(x,y, label = label)
    if title != None:
      plt.title('\n'+title+'\n', c = 'r', fontsize = fontsize)
    plt.ylabel('\nvalue\n',fontsize = fontsize, c = 'b')
    plt.xlabel('\ntime\n',fontsize = fontsize, c = 'b')
    if label != None:
      plt.legend(fontsize=fontsize)
    plt.grid(True);
  else:
    plt.figure(figsize = figsize)
    plt.plot(x,y,label = label)
    if title != None:
      plt.title('\n'+title+'\n', c = 'r', fontsize = fontsize)
    plt.ylabel('\nvalue\n',fontsize = fontsize, c = 'b')
    plt.xlabel('\ntime\n',fontsize = fontsize, c = 'b')
    if label != None:
      plt.legend(fontsize=fontsize)
    plt.grid(True);
    
    
import random
import os
import matplotlib.image as mpimg

def plot_random_image_from_dir(dir_name, num_classes = 2, num_samples = 4, figsize = (15,5)):

  """
  Objetive 
  ---------
  function that return images from directory with specifiec number of random samples and random classes
  
  Parameters 
  ----------
  dir_name : name or path of the directory

  num_classes : numeber of random classes to be displayed, default 2,
  
  num_samples : number of sample from each class to be diplayed, default 4, 
  
  figsize : overall figure size of the plot of 1 class, default (15,5)

  """
  classes = os.listdir(dir_name)
  classes = [x for x in classes if x.find('.') < 0]
  random_classes = random.sample(classes,num_classes )

  for sel_class in random_classes:

    sel_class_path = dir_name+'/'+sel_class
    images = os.listdir(sel_class_path)
    num_images = len(images)

    random_images_ids = [random.choice(range(0, num_images)) for _ in range(0, num_samples)]
    random_images = [images[x] for x in random_images_ids]
    random_image_paths = [sel_class_path+'/'+(x) for x in random_images] 
    
    plt.figure(figsize = figsize)
    plt.suptitle(f'\nclass name : '+sel_class)
    for ind,img_path in enumerate(random_image_paths):
      plt.subplot(1,num_samples,ind+1)
      image_value = mpimg.imread(img_path)
      plt.imshow(image_value)
      plt.axis(False)
      image_shape = image_value.shape
      plt.title(f'shape : {image_shape}');
      
      
## Create Mode checkpoints
import tensorflow as tf  
def create_model_checkpoint(model_name, checkpoint_dir = 'checkpoints', monitor = 'val_accuracy', save_best_only = True,save_as_h5 = True):
  
  """
  Objective 
  ---------
  return the callback object in the .h5 format
  
  Parameters
  ----------
  model_name : Nam of the model being fit
  checkpoint dir : Name of the checkpoint directory, default 'checkpoints'
  monitor : based on the what criteria, one epoch to be selected, default 'val_accuracy'
  """
    # Define the checkpoint path and callback
  if save_as_h5:
    checkpoint_path = f"{checkpoint_dir}/{model_name}.h5"
  else:
    checkpoint_path = f"{checkpoint_dir}/{model_name}"
    
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=False,
                                                             monitor = monitor,
                                                             save_best_only = save_best_only,
                                                             verbose=0)
  return checkpoint_callback



import tensorflow as tf

def reduce_learning_rate_callback(monitor='val_accuracy',patience = 2,factor=0.1):
  
  """
  Objective 
  ---------
  Reduce the learning rate on the fly during traing if a given metric is not improving
  
  Parameters
  ----------
  monitor : The metric to be monitored , default = 'val_accuracy',
  patience : Number epohs the system waits for the metric to improve, default = 2,
  factor : The factor by which the learning_rate to be reduced, defualt = 0.1
  
  """
  return tf.keras.callbacks.ReduceLROnPlateau(monitor = monitor,
                                                patience = patience,
                                                factor = factor,
                                                verbose=1,
                                                min_lr=0 )

import gc
import tensorflow as tf
def clear_memory_tensorflow_session():
  """
  Objetive 
  --------
  Function to clear redundant variables from RAM after traing a model
  which make the more RAM available for the other models to contiue train on...
  """
  tf.keras.backend.clear_session()
  gc.collect()
  
  
import splitfolders
def split_directories(original_dir,output_dir,ratio = (.8,.2)):

    """
    Objective
    ---------
    Splits a file directory into train, val, test sub directories
    You can split into train-val-test or tran-val as well based on the number of ratios provided
    if it is 3 ratios -- > train, val, test
    if it is 2 ratio ---> train & val only
    random state is 42.

    Parameters
    ----------
    orinal_dir : path to the original directory where the files are present
    output_dir : name of the new director to be created, inside of which the split directories will be created
    ratio : the split ratios to each directories. default - > (.8,.2). this tuple can have 2 or 3 elements bases on the context of the split

    """
    splitfolders.ratio(original_dir,
                     output = output_dir,
                     seed = 42,
                     ratio = ratio,
                     group_prefix=None,
                     move=False
                   )
    
import shutil
import os

def remove_directory(dir_path):

  """
  Objective
  -----------
  Removes a directory/sub directory 

  Parameters
  ----------
  dir_path : path of the directory to be removed
  """
  try:
      shutil.rmtree(dir_path)
      
      print(f"Directory '{dir_path}' deleted successfully.\n")
      parent_dir = '/'.join(dir_path.split('/')[:-1])
      remaining_dirs = [x for x in os.listdir(parent_dir) if x.find('.') < 0]
      remaining_dirs_count = len(remaining_dirs)
      print(f'count of remianing sub directories in {parent_dir} : {remaining_dirs_count}\n')
      print(f'list of remianing sub directories in {parent_dir} is\n\n{remaining_dirs}')

  except OSError as e:
      print(f"Error deleting directory '{dir_path}': {e}")
      
import random
import os
import pandas as pd
      
def read_text_file_classes(dir,classes = None, extension = '.txt', shuffle = True):

  """
  Function that reads a text file from directory and retuns a tuple contains lists of features and labels in order
  --> The function shuffle the features and correspinding labels together @ at provided random state 

  dir : The directory path where the class directories are present
  classes : The list of required class names, default None -- > considers all sub directories in dir
  extension : the text file extension, default is .txt
  shuffle : boolean , shuffle at a random state of 42, default True

  """
  features, labels = [], []

  classes = [x for x in os.listdir(dir) if x.find('.') < 0] if classes == None else classes
  
  for cls in classes:
      cls_path = os.path.join(dir, cls)
      label = cls
      print(f'reading <{extension}> files in t: {cls_path}')

      for filename in os.listdir(cls_path):
          if not filename.endswith(extension):
              continue
          filepath = os.path.join(cls_path, filename)

          with open(filepath, 'r') as f:
              feature = f.read()
              features.append(feature)
              labels.append(label)
  
  if shuffle:  
    zipped = list(zip(features, labels))  # zip the two lists together
    random.seed(42) # keeping the seed for the reproducability of the code
    random.shuffle(zipped)  # shuffle the zipped list
    features, labels = zip(*zipped) # unzip the shuffled list into two separate lists
    features, labels = list(features), list(labels) # convreting the tuple output to list

  print(f'\nfeatures found : {len(features)}')
  print(f'lables found : {len(classes)}')
  print(f'\nclasses : {pd.Series(labels).value_counts(normalize = True).index.values}')
  print(f'balance : {pd.Series(labels).value_counts(normalize = True).values}')

  return features,labels


import string
import tensorflow as tf

def text_standardization(text, to_lower=True, rm_punctuations=True, rm_html=True, rm_newline=True):

  """
  Objective
  ---------
  Function that takes the raw text process and return the standardized text for natural language processing task

  Note : It is fuction need to map into each elements of an iterative object

  example --> train_sentences = [text_standardization(x) for x in train_sentences]

  Parameters
  ---------
  text : text to standardize
  to_lower : lower all the alphabets in the text
  rm_punctuations : bool -- > remove the punctuations
  rm_html : bool -- > remove the thml tages from the text
  rm_newline : bool -- > remove the newline ('\n')

  """
  
  if to_lower:
    x = tf.strings.lower(text)  # convert all the text into lowercase
  if rm_html:
    x = tf.strings.regex_replace(x, '<[^>]*>', '')  # replace the html tags with nothing
  if rm_punctuations:
    x = tf.strings.regex_replace(x, '[' + string.punctuation + ']', '')  # replace the punctuations with nothing
  if rm_newline:
    x = tf.strings.regex_replace(x, '\n', '')  # remove the new line characters
  x = x.numpy().decode('utf-8')  # convert into a numpy value and decode the text into utf-8 format
    
  return x


import tensorflow as tf

def calculate_results_regression(y_true, y_pred):

  """
  Objective
  ---------
  Return the evaluation metrics for regression problem based the prediction

  Parameters
  ---------
  y_true : actual values upon which the prediction is made
  y_pred : predicted value from the model

  """

  # Make sure float32 (for metric calculations)
  
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  
  if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
  
  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy()}


import numpy as np

def windowed_times_series(values, window, horizon = 1):
  
  """
  Objetive
  --------
  Return tuple of windewed data features and labels for the given time series values given the window size and tha horizon

  Parameters
  --------
  window : window size
  horizon : number of labels to be predicted
  
  """
  features, labels = [],[]
  for start in range(0,len(values)):  
    window_frame = values[start:(start + window + horizon)]
    if len(window_frame) == (window + horizon):
      feature = window_frame[:-(horizon)]
      label = window_frame[-(horizon):]
      features.append(list(feature))
      labels.append(list(label))
    else:
      break
  features, labels = np.array(features), np.squeeze(np.array(labels))
  print(f'window size : {window} & horizon_size : {horizon}')
  print(f'number of windows returned : {len(features)}\n')
  
  return features,labels

## function for converting the prefetch dataset to numpy array

def prefetch_dataset_to_array(dataset):

  """
  Returns tuples of 2 numpy arrays, -->(data, labels)

  Objective:
  ---------

  To convert the prefetch dataset (batched/plain) to numpy array to have deeper understanding and preprocessing

  Parameters
  ----------
  dataset : prefetch (batched or plain) dataset needs to be converted, 

  """
  
  data, labels = [],[]
  try:
    # check for batches
    dataset = dataset.unbatch()
    print('Unbatching the dataset')

  except:
    None

  for batch in dataset:
      x, y = batch
      data.append(x.numpy())
      labels.append(y.numpy())

  data = np.array(data)
  labels = np.array(labels)
  print('Numpy arrays returned as --> (data, labels)')
  
  return data,labels
