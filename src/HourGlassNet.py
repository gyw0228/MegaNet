
import argparse
import os
from pycocotools.coco import COCO
import numpy as np
import random
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pylab
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
# from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.layers.python.layers import utils

pylab.rcParams['figure.figsize'] = (10.0, 8.0)


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='/Users/kyle/Repositories/coco')
parser.add_argument('--train_data', default='person_keypoints_train2014')
parser.add_argument('--val_data', default='person_keypoints_val2014')
parser.add_argument('--test_data', default='image_info_test-dev2015')
parser.add_argument('--image_train_dir', default='train2014')
parser.add_argument('--image_val_dir', default='val2014')
parser.add_argument('--image_test_dir', default='test2015')

parser.add_argument('--model_path', default='checkpoints/resnet_v2_50.ckpt', type=str)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--small_dataset', default=True, type=bool)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--decay_rate', default=0.01, type=float)
parser.add_argument('--decay_steps', default=50000, type=float)
parser.add_argument('--checkpoint_every', default=50, type=int) # save checkpoint 5 epochs
parser.add_argument('--summary_every', default=200, type=int) # batches per summary
parser.add_argument('--save_path', default='checkpoints/MegaNet', type=str)
parser.add_argument('--log_path', default='/tmp/KyleNet', type=str)

def check_accuracy(sess, keypoint_accuracy, is_training, dataset_init_op, MAX_BATCHES=50):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    evals = 0
    epoch_kpt_accuracy = 0
    while True and evals < MAX_BATCHES:
        try:
            kpt_acc = sess.run(keypoint_accuracy, {is_training: False})
            epoch_kpt_accuracy += kpt_acc
            evals += 1
        except tf.errors.OutOfRangeError:
            break
    epoch_kpt_accuracy = float(epoch_kpt_accuracy/evals)

    return epoch_kpt_accuracy

def keypoint_SigmoidCrossEntropyLoss(graph, prediction_maps, keypoint_masks, labels, L=5.0, scope="keypointLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    with graph.as_default():
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_maps,labels=keypoint_masks)
        labels = tf.reshape(labels,[-1,1,1,17])
        losses = tf.multiply(losses,labels) # set loss to zero for invalid keypoints (labels=0)
        
        return losses

def keypoint_SoftmaxCrossEntropyLoss(graph, prediction_maps, keypoint_masks, labels, scope="keypointLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    with graph.as_default():
        map_shape = prediction_maps.shape.as_list()
        flat_shape = (-1,1,map_shape[1]*map_shape[2],map_shape[3])
        pred_flat = tf.reshape(prediction_maps, flat_shape)
        masks_flat = tf.reshape(keypoint_masks, flat_shape)
        # softmax over dimension 1
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=masks_flat,logits=pred_flat,dim=2)
        labels = tf.reshape(labels,[-1,1,1,17])
        losses = tf.multiply(losses,labels) # set loss to zero for invalid keypoints (labels=0)
        
        return losses
def mask_SoftmaxCrossEntropyLoss(graph, mask_prediction, mask_true, scope="maskLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    with graph.as_default():
        map_shape = mask_prediction.shape.as_list()
        flat_shape = (-1,1,map_shape[1]*map_shape[2],map_shape[3])
        pred_flat = tf.reshape(mask_prediction, flat_shape)
        masks_flat = tf.reshape(mask_true, flat_shape)
        # softmax over dimension 1
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=masks_flat,logits=pred_flat,dim=2)
        
        return losses

def keypoint_TargetedCrossEntropyLoss(graph, prediction_maps, keypoint_masks, labels, L=5.0, scope="keypointLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    with graph.as_default():
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_maps,labels=keypoint_masks)
        losses = tf.multiply(losses, keypoint_masks) # Constrain loss to correct prediction region
        labels = tf.reshape(labels,[-1,1,1,17])
        losses = tf.multiply(losses,labels) # set loss to zero for invalid keypoints (labels=0)
        
        return losses


def keypoint_SquaredErrorLoss(graph, prediction_maps, keypoint_masks, labels, L=5.0, scope="keypointLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    with graph.as_default():
        with tf.variable_scope(scope):
            losses = tf.squared_difference(prediction_maps,keypoint_masks)
            labels = tf.reshape(labels,[-1,1,1,17])
            losses = tf.multiply(losses,labels) # set loss to zero for invalid keypoints (labels=0)
            
            return losses

def KeypointPrediction(graph, pred_masks, d, scope='KeypointPrediction'):
    """
    Input: Keypoint "Heatmap" Tensor
    Output: Keypoint coordinates in tensor form
    """
    with graph.as_default():        
        with tf.variable_scope(scope):
            x = tf.reshape(tf.linspace(0.5,d-0.5,d),[1,d,1,1])
            pred = tf.multiply(pred_masks, tf.to_float(tf.greater_equal(pred_masks,0.5)))
            pred_i = tf.reduce_sum(tf.multiply(pred, x),axis=[1,2])/tf.reduce_sum(pred,axis=[1,2])
            pred_j = tf.reduce_sum(tf.multiply(pred, tf.transpose(x,(0,2,1,3))),axis=[1,2])/tf.reduce_sum(pred,axis=[1,2])
            pred_pts = tf.stack([pred_j,pred_i],axis=1)
            pred_pts = tf.expand_dims(pred_pts,axis=1)
            return pred_pts

def keypointPredictionAccuracy(graph, pred_pts, true_pts, labels, threshold, scope='KeypointPrediction'):
    """
    Accuracy is a boolean: 1 if ||pred_pt-true_pt||^2 < threshold^2, 0 otherwise
    """
    with graph.as_default():
        with tf.variable_scope(scope):
            error = tf.multiply(tf.square(tf.subtract(pred_pts, true_pts)), tf.to_float(tf.greater_equal(labels, 1)))
            accuracy = tf.reduce_mean(tf.to_float(tf.less(error,tf.square(threshold))))
            return accuracy

def MaskAccuracy(graph, pred_mask, true_mask):
    with graph.as_default():        
        overlap = tf.reduce_sum(tf.multiply(tf.to_float(pred_mask),tf.to_float(true_mask)),axis=[1,2,3])
        score1 = tf.divide(overlap, tf.reduce_sum(tf.to_float(pred_mask),axis=[1,2,3]))
        score2 = tf.divide(overlap, tf.reduce_sum(tf.to_float(true_mask),axis=[1,2,3]))
        accuracy = tf.minimum(score1,score2)
        return tf.reduce_mean(accuracy)


# Initialize Dataset
def get_data(base_dir,image_dir,ann_file):
    image_path = '{}/images/{}'.format(base_dir,image_dir)
    ann_path='{}/annotations/{}.json'.format(base_dir,ann_file)

    return image_path, ann_path
    
# define the path to the annotation file corresponding to the images you want to work with
#######################################################
#### VISUALIZATION TOOLS - WEIGHTS AND ACTIVATIONS ####
#######################################################
def highestPrimeFactorization(n):    
    return [(i, n//i) for i in range(1, int(n**0.5) + 1) if n % i == 0][-1] 

def getFilterImage(filters):
    """
    Takes as input a filter bank of size (1, H, W, C, D)
    Returns: a tensor of size (1, sqrt(D)*H, sqrt(D)*H, C)
    (This only really works for the first layer of filters with an image as input)
    """
    padded_filters = tf.pad(filters,tf.constant([[0,0],[1,0],[1,0],[0,0],[0,0]]),'CONSTANT')
    filter_list = tf.unstack(padded_filters,axis=4)
    H,W = highestPrimeFactorization(len(filter_list))
    weight_strips = [tf.concat(filter_list[8*i:8*(i+1)],axis=1) for i in range(W)]
    weight_image = tf.concat(weight_strips,axis=2)
    return weight_image

def getActivationImage(activations, scale_up=False):
    """
    Tiles an activation map into a square grayscale image
    Takes as input an activation map of size (N, H, W, D)
    Returns: a tensor of size (N, sqrt(D)*H, sqrt(D)*H, 1)
    """
    padded_activations = tf.pad(activations,tf.constant([[0,0],[1,0],[1,0],[0,0]]),'CONSTANT')
    expanded_activations = tf.expand_dims(padded_activations,axis=3)
    activations_list = tf.unstack(expanded_activations,axis=4)
    H,W = highestPrimeFactorization(len(activations_list))
    activation_strips = [tf.concat(activations_list[H*i:H*(i+1)],axis=1) for i in range(W)]
    activation_image = tf.concat(activation_strips,axis=2)            
    if scale_up:
        activation_image = tf.divide(activation_image, tf.reduce_max(activation_image))

    return activation_image



#######################################################
########### NETWORK DEFINITION FUNCTION(S) ############
#######################################################
def HourGlassNet(graph, inputs=None, num_levels=5, base_filters = 64, scalar_summary_list=None, image_summary_list=None, histogram_summary_list=None):
    """
    Returns:
    - net 
    - summary lists
    """
    with graph.as_default():
        # Initialize summaries
        if scalar_summary_list is None:
            scalar_summary_list = []
        if image_summary_list is None:
            image_summary_list = []
        if histogram_summary_list is None:
            histogram_summary_list = []
            
        head = {}
        with tf.variable_scope('Hourglass'):
            with tf.variable_scope('base'):
                base = tf.layers.conv2d(inputs, base_filters, (3,3),strides=(1,1),padding='SAME',bias_initializer=tf.constant_initializer(0.01))
                histogram_summary_list.append(tf.summary.histogram('base', base))
                image_summary_list.append(tf.summary.image('base',getActivationImage(base, scale_up=True)))
                base = tf.layers.batch_normalization(base,axis=3)
                base = tf.nn.relu(base)

            for level in range(num_levels):
                with tf.variable_scope('level_{}'.format(level+1)):
                    bridge_filters = base_filters*2**level
                    # Base - decrease size by factor of 2 for n
                    base = tf.layers.conv2d(base,bridge_filters,(3,3),strides=(2,2),padding='SAME',bias_initializer=tf.constant_initializer(0.1))
                    histogram_summary_list.append(tf.summary.histogram('level_{}_base'.format(level+1), base))
                    base = tf.layers.batch_normalization(base,axis=3)
                    base = tf.nn.relu(base)

                    # Bridge - maintain constant size
                    bridge = base
                    for i in range(num_levels - level):
                        bridge = tf.layers.conv2d(bridge,bridge_filters,(3,3),strides=(1,1),padding='SAME',bias_initializer=tf.constant_initializer(0.1))
                        histogram_summary_list.append(tf.summary.histogram('level_{}_bridge_{}'.format(level+1,i+1), bridge))
                        bridge = tf.layers.batch_normalization(bridge,axis=3)
                        bridge = tf.nn.relu(bridge)
                    head[level] = bridge
                    image_summary_list.append(tf.summary.image('bridge_{}'.format(level+1), getActivationImage(bridge, scale_up=True)))
                    
            for level in reversed(range(num_levels)):
                # resize_bilinear or upconv?
                # output = tf.image.resize_bilinear(ouput,size=2*output.shape[1:2])
                out_filters = int(base_filters*2**(level-1))

                output = tf.layers.conv2d_transpose(head[level],out_filters,(3,3),(2,2),padding='SAME',bias_initializer=tf.constant_initializer(0.1))
                histogram_summary_list.append(tf.summary.histogram('level_{}_out'.format(level+1), output))
                output = tf.layers.batch_normalization(output,axis=3) # HERE OR AFTER CONCAT???
                output = tf.nn.relu(output) # HERE OR AFTER CONCAT???

                if level > 0:
                    head[level-1] = tf.concat([head[level-1],output],axis=3)

            variables = tf.trainable_variables()
            endpoints = {}
            for v in variables:
                endpoints[v.name] = v

            image_summary_list.append(tf.summary.image('layer1_conv', getFilterImage(tf.expand_dims(endpoints['network/Hourglass/base/conv2d/kernel:0'],0))))
                
            return head[0], scalar_summary_list, image_summary_list, histogram_summary_list



def main(args):
    ######################## Data Path ########################
    train_img_path, train_ann_path = get_data(args.base_dir,args.image_train_dir,args.train_data)
    val_img_path, val_ann_path = get_data(args.base_dir,args.image_val_dir,args.val_data)
    # initialize a coco object
    print("Initializing COCO objects to extract training and validation datasets...\n")
    train_coco = COCO(train_ann_path)
    val_coco = COCO(val_ann_path)
    # get all images containing the 'person' category
    train_catIds = train_coco.getCatIds(catNms=['person'])
    train_imgIds = train_coco.getImgIds(catIds=train_catIds)
    val_catIds = val_coco.getCatIds(catNms=['person'])
    val_imgIds = val_coco.getImgIds(catIds=val_catIds)
    # Just for dealing with the images on my computer (not necessary when working with the whole dataset)
    if args.small_dataset:
        train_catIds = train_catIds[0:30]
        train_imgIds = train_imgIds[0:30]
        val_catIds = val_catIds[0:30]
        val_imgIds = val_imgIds[0:30]

    graph = tf.Graph()
    with graph.as_default():
        
        #######################################################
        ############### VARIOUS HYPER-PARAMETERS ##############
        #######################################################

        NUM_KEYPOINTS = 17
        BATCH_SIZE = args.batch_size
        L = 5.0 # keypoint effective radius
        D = 224 # image height and width
        d = 112 # evaluation height and width (for mask and keypoint masks)

        MASK_THRESHOLD = 0.5 # threshold for on/off prediction (in mask and keypoint masks)
        KP_THRESHOLD = 0.5 # threshold for on/off prediction (in mask and keypoint masks)
        KP_DISTANCE_THRESHOLD = 5.0 # threshold for determining if a keypoint estimate is accurate
        X_INIT = tf.contrib.layers.xavier_initializer_conv2d() # xavier initializer for head architecture
        learning_rate1 = args.learning_rate1
        learning_rate2 = args.learning_rate2

        #######################################################
        ##### PRE-PROCESSING AND DATASET EXTRACTION TOOLS #####
        #######################################################
        def extract_annotations_train(filename, imgID, coco=train_coco):
            anns = coco.loadAnns(coco.getAnnIds(imgID,catIds=[1],iscrowd=None))
            ann = max([ann for ann in anns], key=lambda item:item['area']) # extract annotation for biggest instance
            bbox = np.array(np.floor(ann['bbox']),dtype=int)
            keypoints = np.reshape(ann['keypoints'],(-1,3))
            mask = coco.annToMask(ann)
            
            return filename, bbox, keypoints, mask

        def extract_annotations_val(filename, imgID, coco=val_coco):
            anns = coco.loadAnns(coco.getAnnIds(imgID,catIds=[1],iscrowd=None))
            ann = max([ann for ann in anns], key=lambda item:item['area']) # extract annotation for biggest instance
            bbox = np.array(np.floor(ann['bbox']),dtype=int)
            keypoints = np.reshape(ann['keypoints'],(-1,3))
            mask = coco.annToMask(ann)
            
            return filename, bbox, keypoints, mask
        
        def preprocess_image_tf(filename, bbox_tensor, keypoints_tensor, mask, D=D):
            """
            Returns:
            resized_image (N,D,D,3) - cropped, padded (if needed), scaled to square image of size D
            resized_mask (N,D,D,1) - cropped, padded (if needed), scaled to square mask of size D
            pts (N,2,17) - keypoint coordinates (i,j) scaled to match up with resized_image
            labels (N,1,17) - values corresponding to pts: {0: invalid, 1:occluded, 2:valid}
            """
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            
            # subtract mean
            image = tf.subtract(image, tf.reduce_mean(image))

            mask = tf.transpose([mask],[1,2,0])
            bbox_tensor = tf.to_float(bbox_tensor)
            keypoints_tensor = tf.to_float(keypoints_tensor)

            sideLength = tf.reduce_max(bbox_tensor[2:],axis=0)
            centerX = tf.floor(bbox_tensor[0] + tf.divide(bbox_tensor[2],tf.constant(2.0)))
            centerY = tf.floor(bbox_tensor[1] + tf.divide(bbox_tensor[3],tf.constant(2.0)))
            center = tf.stack([centerX,centerY])

            corner1 = tf.to_int32(tf.minimum(tf.maximum(tf.subtract(center, tf.divide(sideLength,tf.constant(2.0))),0),
                                tf.reverse(tf.to_float(tf.shape(image)[:2]),tf.constant([0]))))
            corner2 = tf.to_int32(tf.minimum(tf.maximum(tf.add(center, tf.divide(sideLength,tf.constant(2.0))),0),
                                tf.reverse(tf.to_float(tf.shape(image)[:2]),tf.constant([0]))))
            i_shape = tf.subtract(corner2,corner1)
            d_shape = tf.subtract(tf.to_int32(sideLength),i_shape)

            scale = tf.divide(tf.constant(D,tf.float32), sideLength)
            cropped_image = tf.image.crop_to_bounding_box(image,corner1[1],corner1[0],
                                                        tf.subtract(corner2,corner1)[1],tf.subtract(corner2,corner1)[0])
            cropped_mask = tf.image.crop_to_bounding_box(mask,corner1[1],corner1[0],
                                                        tf.subtract(corner2,corner1)[1],tf.subtract(corner2,corner1)[0])

            dX = tf.floor(tf.divide(d_shape,tf.constant(2)))
            dY = tf.ceil(tf.divide(d_shape,tf.constant(2)))

            pts, labels = tf.split(keypoints_tensor,[2,1],axis=1)
            pts = tf.subtract(pts,tf.to_float(corner1)) # shift keypoints
            pts = tf.add(pts,tf.to_float(dX)) # shift keypoints
            pts = tf.multiply(pts,scale) # scale keypoints

            # set invalid pts to 0
            valid = tf.less(pts,tf.constant(D,tf.float32))
            valid = tf.multiply(tf.to_int32(valid), tf.to_int32(tf.greater(pts,0)))
            pts = tf.multiply(pts,tf.to_float(valid))
            pts = tf.transpose(pts,[1,0])
            labels = tf.transpose(labels,[1,0])
            labels = tf.to_float(tf.greater_equal(labels, 2))

            padded_image = tf.image.pad_to_bounding_box(cropped_image,tf.to_int32(dX[1]),tf.to_int32(dX[0]),
                                                        tf.to_int32(sideLength),tf.to_int32(sideLength))
            padded_mask = tf.image.pad_to_bounding_box(cropped_mask,tf.to_int32(dX[1]),tf.to_int32(dX[0]),
                                                        tf.to_int32(sideLength),tf.to_int32(sideLength))

            resized_image = tf.image.resize_images(padded_image,tf.constant([D,D]),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # resized_image = resized_image - VGG_MEAN
            resized_mask = tf.image.resize_images(padded_mask,tf.constant([D,D]),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return resized_image, resized_mask, pts, labels

        def scaleDownMaskAndKeypoints(image, mask, pts, labels, d=d, D=D):
            mask = tf.image.resize_images(mask,tf.constant([d,d]),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            pts = tf.multiply(pts,tf.constant(d/D))
            return image, mask, pts, labels
        
        def generate_keypoint_masks(image, mask, keypoints, labels, d=d, D=D, L=L):
            X, Y = tf.meshgrid(tf.linspace(0.0,d,d),tf.linspace(0.0,d,d))
            X = tf.reshape(X,[d,d,1])
            Y = tf.reshape(Y,[d,d,1])
            X_stack = tf.tile(X,tf.constant([1,1,17],dtype=tf.int32))
            Y_stack = tf.tile(Y,tf.constant([1,1,17],dtype=tf.int32))

            pts = tf.reshape(keypoints,[1,2,17])
            ptsX, ptsY = tf.split(pts,[1,1],axis=1)
            d1 = tf.square(tf.subtract(X_stack,ptsX))
            d2 = tf.square(tf.subtract(Y_stack,ptsY))

            pt_masks = tf.multiply(tf.divide(tf.constant(1.0),tf.add(d1,d2)+L),L)
            return image, mask, pt_masks, pts, labels
        def generate_one_hot_keypoint_masks(image, mask, keypoints, labels, d=d):
            pts = tf.reshape(keypoints,[1,2,17])
            indices = tf.to_int32(pts)
            kp_mask1 = tf.one_hot(depth=d,indices=indices[:,1,:],axis=0)
            kp_mask2 = tf.one_hot(depth=d,indices=indices[:,0,:],axis=1)
            kp_masks = tf.matmul(tf.transpose(kp_mask1,(2,0,1)),tf.transpose(kp_mask2,(2,0,1)))
            kp_masks = tf.transpose(kp_masks,(1,2,0))
            return image, mask, kp_masks, pts, labels

        
        def softmax_keypoint_masks(kpt_masks, d=d):
            return tf.reshape(tf.nn.softmax(tf.reshape(kpt_masks, [-1,d**2,17]),dim=1),[-1,d,d,17])
        def bilinear_filter(channels_in,channels_out):
            f = tf.multiply(tf.constant([0.5, 1.0, 0.5],shape=[3,1]),tf.constant([0.5, 1.0, 0.5],shape=[1,3]))
            f = tf.stack([f for i in range(channels_out)],axis=2)
            f = tf.stack([f for i in range(channels_in)],axis=3)
            return f

         
        #######################################################
        ################## SUMMARY DICTIONARY #################
        #######################################################

        image_summary_list = []
        scalar_summary_list = []
        histogram_summary_list = []

        #######################################################
        ################### PREPARE DATASET ###################
        #######################################################
        print("Initializing Dataset...\n")
        with tf.variable_scope('DataSet'):
            ################### TRAIN DATASET ###################
            train_filenames = tf.constant(['{}/COCO_train2014_{:0>12}.jpg'.format(train_img_path, imgID) for imgID in train_imgIds])
            train_imgID_tensor = tf.constant(train_imgIds)
            train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_filenames, train_imgID_tensor))
            train_dataset = train_dataset.map(
                lambda filename, imgID: tf.py_func(extract_annotations_train, [filename, imgID], [filename.dtype, tf.int64, tf.int64, tf.uint8]))
            train_dataset = train_dataset.map(preprocess_image_tf)
            train_dataset = train_dataset.map(scaleDownMaskAndKeypoints)
            train_dataset = train_dataset.map(generate_one_hot_keypoint_masks)
            train_dataset = train_dataset.shuffle(buffer_size=10000)
            train_dataset = train_dataset.batch(BATCH_SIZE)

            #################### VAL DATASET ####################
            val_filenames = tf.constant(['{}/COCO_val2014_{:0>12}.jpg'.format(val_img_path, imgID) for imgID in val_imgIds])
            val_imgID_tensor = tf.constant(val_imgIds)
            val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_imgID_tensor))
            val_dataset = val_dataset.map(
                lambda filename, imgID: tf.py_func(extract_annotations_val,[filename, imgID],[filename.dtype, tf.int64, tf.int64, tf.uint8]))
            val_dataset = val_dataset.map(preprocess_image_tf)
            val_dataset = val_dataset.map(scaleDownMaskAndKeypoints)
            val_dataset = val_dataset.map(generate_one_hot_keypoint_masks)
            val_dataset = val_dataset.shuffle(buffer_size=10000)
            val_dataset = val_dataset.batch(BATCH_SIZE)

            iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

            images, masks, kpt_masks, pts, labels = iterator.get_next()
            train_init_op = iterator.make_initializer(train_dataset)
            val_init_op = iterator.make_initializer(val_dataset)

            image_summary_list.append(tf.summary.image('keypoint masks', getActivationImage(kpt_masks, scale_up=True)))
            image_summary_list.append(tf.summary.image('input images', images))
        
        #######################################################
        ##################### BUILD GRAPH #####################
        #######################################################

        is_training = tf.placeholder(tf.bool)

        # --------------------------------------------------- #
        # --------------- "Head" Architecture --------------- #
        # --------------------------------------------------- #
        print("Defining Network architecture...\n")
        HEAD_SCOPE='network'
        with tf.variable_scope(HEAD_SCOPE):
            net = tf.layers.conv2d(images,64,(3,3),(1,1),padding='SAME',bias_initializer=tf.constant_initializer(0.1))
            net = tf.layers.batch_normalization(net,axis=3)
            net = tf.nn.relu(net)

            net, scalar_summary_list, image_summary_list, histogram_summary_list = HourGlassNet(
                graph,
                inputs=images,
                num_levels=4,
                base_filters = 64,
                scalar_summary_list=scalar_summary_list,
                image_summary_list=image_summary_list,
                histogram_summary_list=histogram_summary_list)

            logits = tf.layers.conv2d(net,17,(1,1),(1,1),padding='SAME')

        ########## Prediction and Accuracy Checking ########### 

            with tf.variable_scope('predictions'):
                keypoint_mask_predictions = softmax_keypoint_masks(logits, d=d)
                keypoint_predictions = KeypointPrediction(graph,keypoint_mask_predictions,d=d)
                keypoint_accuracy = keypointPredictionAccuracy(graph,keypoint_predictions,pts,labels,threshold=4.0)

                image_summary_list.append(tf.summary.image('Head - keypoint mask prediction', getActivationImage(keypoint_mask_predictions, scale_up=True)))
                for i in [1.0,2.0,3.0,5.0,8.0]:
                    scalar_summary_list.append(tf.summary.scalar(
                        'Head - keypoint accuracy delta={}'.format(i), keypointPredictionAccuracy(graph, keypoint_predictions, pts, labels, i)
                    ))

        #######################################################
        ####################### LOSSES ########################
        #######################################################

            with tf.variable_scope('loss'):
                keypoint_loss = tf.reduce_mean(keypoint_SoftmaxCrossEntropyLoss(graph,logits,kpt_masks,labels))
                total_loss = keypoint_loss

                scalar_summary_list.append(tf.summary.scalar('loss', total_loss))
            
            # Call to initialize Head Variables from scratch
            head_variables = tf.contrib.framework.get_variables(HEAD_SCOPE)
            init_head = tf.variables_initializer(head_variables, 'init_head')

        #######################################################
        ###################### SUMMARIES ######################
        #######################################################

            image_summary = tf.summary.merge(image_summary_list, collections=None, name="Image_Summaries")
            scalar_summary = tf.summary.merge(scalar_summary_list, collections=None, name="Scalar_Summaries")
            histogram_summary = tf.summary.merge(histogram_summary_list, collections=None, name="Histogram_summaries")

            # Saver to save graph to checkpoint
            head_saver = tf.train.Saver(var_list=head_variables, max_to_keep=5)

        #######################################################
        ###################### OPTIMIZERS #####################
        #######################################################

        with tf.variable_scope('Optimizers'):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            # Exponential decay learning schedule
            learning_rate = tf.train.exponential_decay(
                learning_rate=args.learning_rate1, 
                global_step=global_step, 
                decay_steps=args.decay_steps, 
                decay_rate=args.decay_rate, 
                staircase=True
                )
            # begin trainiing with a really low learning rate to avoid killing all of the neurons at the start
            optimizer_1 = tf.train.RMSPropOptimizer(learning_rate=1e-7)
            train_op_1 = optimizer_1.minimize(total_loss, global_step=global_step, var_list=head_variables, gate_gradients=tf.train.RMSPropOptimizer.GATE_NONE)
            
            head_optimizer = tf.train.RMSPropOptimizer(learning_rate)
            head_train_op = head_optimizer.minimize(total_loss, global_step=global_step, var_list=head_variables, gate_gradients=tf.train.RMSPropOptimizer.GATE_NONE)
        
        # RMSProp optimizer uses "slot" variables for maintaining the running average of weight updates. It must therefore be initialized 
        optimizer_variables = tf.contrib.framework.get_variables('Optimizers')
        init_optimizer = tf.variables_initializer(optimizer_variables)

        # Finalize default graph - THIS SEEMS TO PREVENT ADDING A FILEWRITER LATER
        tf.get_default_graph().finalize()

        #######################################################
        ######################## TRAIN ########################
        #######################################################
        with tf.Session(graph=graph) as sess:
            # file writer to save graph for Tensorboard
            log_path = args.log_path
            i = 0
            while True:
                if os.path.isdir('{}/{}'.format(log_path,i)):
                    i += 1
                else:
                    log_path = '{}/{}'.format(log_path,i)
                    break

            file_writer = tf.summary.FileWriter(log_path)
            file_writer.add_graph(sess.graph)
            print("Initializing backbone variables...")
            # init_fn(sess) # pretrained backbone variables
            print("Initializing head variables...")
            sess.run(init_head) # head variables
            print("Initializing optimizer variables...\n")
            sess.run(init_optimizer)

            print("Beginning Training...")

            #############################################################
            ###################### TRAINING ROUND 0 #####################
            #############################################################
            print('### Starting Epoch 0 - burn in with low learning rate')
            sess.run(train_init_op)
            batch = 0
            while True:
                    try:
                        kp_loss, _ = sess.run([keypoint_loss, head_train_op], {is_training: True})
                        print('----- Losses for batch {}: Keypoint Loss: {:0>5}'.format(batch+1, kp_loss))
                        if batch % args.summary_every == 0:
                            image_summ, scalar_summ, histogram_summ = sess.run([image_summary, scalar_summary, histogram_summary],{is_training: False})
                            file_writer.add_summary(image_summ, global_step=tf.train.global_step(sess, global_step))
                            file_writer.add_summary(scalar_summ, global_step=tf.train.global_step(sess, global_step))
                            file_writer.add_summary(histogram_summ, global_step=tf.train.global_step(sess, global_step))
                        else:
                            scalar_summ = sess.run(scalar_summary, {is_training: False})
                            file_writer.add_summary(scalar_summ, global_step=tf.train.global_step(sess, global_step))
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break

            #############################################################
            ###################### TRAINING ROUND 0 #####################
            #############################################################
            for epoch in range(args.num_epochs1):
                # Run an epoch over the training data.
                print('### Starting epoch {}/{} ####################'.format(epoch + 1, args.num_epochs1))
                sess.run(train_init_op) # initialize the iterator with the training set.
                batch = 0
                while True:
                    try:
                        kp_loss, _ = sess.run([keypoint_loss, head_train_op], {is_training: True})
                        print('----- Losses for batch {}: Keypoint Loss: {:0>5}'.format(batch+1, kp_loss))
                        if batch % args.summary_every == 0:
                            image_summ, scalar_summ, histogram_summ = sess.run([image_summary, scalar_summary, histogram_summary],{is_training: False})
                            file_writer.add_summary(image_summ, global_step=tf.train.global_step(sess, global_step))
                            file_writer.add_summary(scalar_summ, global_step=tf.train.global_step(sess, global_step))
                            file_writer.add_summary(histogram_summ, global_step=tf.train.global_step(sess, global_step))
                        else:
                            scalar_summ = sess.run(scalar_summary, {is_training: False})
                            file_writer.add_summary(scalar_summ, global_step=tf.train.global_step(sess, global_step))
                        batch += 1
                    except tf.errors.OutOfRangeError:
                        break

                if epoch % args.checkpoint_every == 0:
                    head_saver.save(sess, args.save_path, global_step=tf.train.global_step(sess, global_step))

                # # Check accuracy on the train and val sets every epoch.
                print("DO NOT PANIC - CHECKING ACCURACY")
                kpt_train_acc = check_accuracy(sess, keypoint_accuracy, is_training, train_init_op)
                kpt_val_acc = check_accuracy(sess, keypoint_accuracy, is_training, val_init_op)
                print('Train accuracy ---- Keypoints_5: {:0>5}'.format(kpt_train_acc))
                print('  Val accuracy ---- Keypoints_5: {:0>5}'.format(kpt_val_acc))

            print("Finished")
            return



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)