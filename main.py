import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# Give the path to the cloned/downloaded tensorflow's 'models' directory
sys.path.append("PATH_TO_TENSORFLOW_MODELS_DIR/models/research/")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#GO TO models/research/object_detection/utils/ and replace the visualization_utils.py file 
#with the one included in this repository.
#this is a necessary step for running the return_coordinates function
from object_detection.utils import visualization_utils as vis_util
#print(vis_util.__file__)

#path to the dir where script_gen.py and preprocess.py files are stored
sys.path.append('PATH_TO_DIR/')
from script_gen import generate_html
from preprocess import preprocessing

import webbrowser

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#link for downloading the trained model:
#https://www.dropbox.com/sh/r7m3p0qikumtjuc/AABKP8kGBUzE8-pJo-WqGWD9a?dl=0
#also present in the model_files/how_to_download_trained_model.txt file 
#place the 'frozen_inference_graph_816.pb' file in the model_file dir after downloading
PATH_TO_FROZEN_GRAPH = 'PATH_TO_model_files_DIR/frozen_inference_graph_816.pb'

#give the path to label.pbtxt file
#it can be found in model_files directory
PATH_TO_LABELS = 'PATH_TO_model_files_DIR/labels.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, -1)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (950, 1000)

coordinates=0


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])


        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
        #DETECTION = output_dict['detection_masks']
  
  return output_dict


def cv2_to_pil(img): #Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def captured_image(img):
      print("captured")
      cv2.imshow('label', img)
      cv2.waitKey()


#sort the list based on y coordinate
def sort_list_y(line):    
    line.sort(key = lambda y: y[2], reverse = False)  
    #print('before changing the value of y',line)
    k=1
    for i in range(len(line)-1):
      #compare current with next
      #sorted in ascending, ie revrese sub
      if((int(line[i+1][2]) - int(line[i][2]))>30):
        #assign k to current 
        line[i][2]=k
        #increnent k
        k=k+1
      else:
        #sub is less than equal to 10, ie same line
        #assign k to current, do not increment k
        line[i][2]=k
    #end of loop, assign number to last element
    line[len(line)-1][2]=k

    #print('after updating the order of line', line)
    return line 


#sort the list based on x coordinate
def sort_list_x(line):
  #print(line)
  t=[]
  new_list=[]
  j=0
  for i in range(len(line)):
    #gather together the elements of the same row
    while(j<len(line)):
      if(line[j][2]==i+1):
        #belong in the same line
        t.append(line[j])
        j=j+1
      else:
        #j=j-1
        break
    #out of loop: all the elements of the same line are in listt
    #sort the elements of the same row in ascending order
    t.sort(key = lambda x: x[1], reverse = False) 
    #append this to the new list is t is not empty
    if(t):
      new_list.append(t)
    t=[]
  #print('\nafter sorting',new_list)

  return new_list 


#mehtod for writing temp file which contains detected html element desctiption
def write_temp(line):
  #define a path to store the detected element desctiption text file
  #this file enables the generate_html function to create html code  
  path='PATH_TO_DIR_CONTAINING_temp.txt/_FILE'
  temp=open(path+'temp.txt','w')
  #convert multidim list to one dim
  flat = [x for sublist in line for x in sublist]
  flat = [x for sublist in flat for x in sublist]
  #create lists of 3 elements each inside flat list
  count=0
  k=[]
  f=[]
  for i in range(len(flat)):
    k.append(flat[i])
    count=count+1
    if(int(count%3)==0):
      f.append(k)
      k=[]

  flat=f
  prev=flat[0][2]
  for i in range(len(flat)):
    if(flat[i][2]>prev):
      #the line has changed, add break
      temp.write('BREAK')
      temp.write('\n')
      temp.write(str(flat[i][0]))
      temp.write(',')
      temp.write(str(flat[i][1]))
      temp.write(',')
      temp.write(str(flat[i][2]))
      temp.write('\n')
      prev=flat[i][2]
    else:
      temp.write(str(flat[i][0]))
      temp.write(',')
      temp.write(str(flat[i][1]))
      temp.write(',')
      temp.write(str(flat[i][2]))
      temp.write('\n')



#create temporary file for the detected elements
def generate_temp(coordinates):
  line=[]
  #extract the info for all the elements
  for c in coordinates:
    xmin = c[2] 
    ymin = c[0]
    xmax = c[3]
    ymax = c[1]
    name = c[4][0].split(':')[0]  
    #calculate the centre points for detected elements
    centre_x = int((xmin+xmax)/2)
    centre_y = int((ymin+ymax)/2)
    #add it to the list
    line.append([name, centre_x, centre_y])
  #sort the list in ascending order based on y
  line=sort_list_y(line)
  #sort the list in ascending order baseed on x
  line=sort_list_x(line)
  #write all the info in temp file
  write_temp(line)
  #generate html code
  generate_html()


def processImage(image): 
    image = Image.open(image)
    width = 950
    height = 1000
    dim = (width, height) 
    image = image.resize((width, height), Image.NEAREST)
    
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    image_np = cv2_to_pil(image_np)
    image_np = load_image_into_numpy_array(image_np)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        max_boxes_to_draw=60,
        use_normalized_coordinates=True,
        min_score_thresh=0.50,
        line_thickness=5)

    #get the coordinates of the detected elements
    coordinates=vis_util.return_coordinates(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        line_thickness=5,
        min_score_thresh=0.50)
    
    #if any objects are detected, generate the temp file
    if(coordinates):
      generate_temp(coordinates)
    
    dim=(950, 1000)
    resized = cv2.resize(image_np, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('output_image',resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #the image has been processed
    #delete the temporary processes file if exists
    path='/home/mohak/Music/implementation/'
    os.remove(path+'temp.jpg')
    #run the generated html code in browser as a webpage
    #give the complete path to the generated html file
    file_path = 'PATH_TO_DIR/generated_code.html'
    new = 2
    webbrowser.open(file_path,new=new)


#For running on local testImages
#path of the image using which the html code is to be generated
path='COMPLETE_PATH_TO_INPUT_IMAGE'

processed_image=preprocessing(path)
processImage(processed_image)

