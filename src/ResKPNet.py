
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

parser.add_argument('--model_path', default='checkpoints/vgg_16.ckpt', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

HEAD_SCOPE = 'Head'

def keypoint_CrossEntropyLoss(prediction_maps, keypoint_masks, labels, L=5.0, scope="keypointLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_maps,labels=keypoint_masks)
    labels = tf.reshape(labels,[-1,1,1,17])
    losses = tf.multiply(losses,labels) # set loss to zero for invalid keypoints (labels=0)
    
    return losses


def keypoint_SquaredErrorLoss(prediction_maps, keypoint_masks, labels, L=5.0, scope="keypointLoss"):
    """
    heat_maps = predictions from network
    keypoints (N,17,2) = actual keypoint locations
    labels (N,17,1) = 0 if invalid, 1 if occluded, 2 if valid
    """
    losses = tf.squared_difference(prediction_maps,keypoint_masks)
    labels = tf.reshape(labels,[-1,1,1,17])
    losses = tf.multiply(losses,labels) # set loss to zero for invalid keypoints (labels=0)
    
    return losses


# Initialize Dataset
def get_data(base_dir,image_dir,ann_file):
    image_path = '{}/images/{}'.format(base_dir,image_dir)
    ann_path='{}/annotations/{}.json'.format(base_dir,ann_file)

    return image_path, ann_path
    
# define the path to the annotation file corresponding to the images you want to work with

def main(args):
    ######################## Data Path ########################
    # baseDir = args.base_dir
    # trainData = args.train_data
    # valData = args.val_data
    # testData = args.test_data
    # imageTrainDir = args.image_train_dir
    # imageValDir = args.image_val_dir
    # imageTestDir = args.image_test_dir

    train_img_path, train_ann_path = get_data(args.base_dir,args.image_train_dir,args.train_data)
    val_img_path, val_ann_path = get_data(args.base_dir,args.image_val_dir,args.val_data)
    # initialize a coco object
    coco = COCO(train_ann_path)
    # get all images containing the 'person' category
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    # Just for dealing with the images on my computer (not necessary when working with the whole dataset)
    catIds = imgIds[0:30]
    imgIds = imgIds[0:30]

    graph = tf.Graph()
    with graph.as_default():
        
        NUM_KEYPOINTS = 17
        BATCH_SIZE = 10
        L = 10.0 # keypoint effective radius
        D = 225 # image height and width
        
        #######################################################
        #### VISUALIZATION TOOLS - WEIGHTS AND ACTIVATIONS ####
        #######################################################
        def highestPrimeFactorization(n):    
            return [(i, n//i) for i in range(1, int(n**0.5) + 1) if n % i == 0][-1] 

        def getFilterImage(filters):
            """
            Takes as input a filter bank of size (1, H, W, C, D)
            Returns: a tensor of size (1, sqrt(D)*H, sqrt(D)*H, C)
            (This only really works for the first layer of filtes with an image as input)
            """
            padded_filters = tf.pad(filters,tf.constant([[0,0],[1,0],[1,0],[0,0],[0,0]]),'CONSTANT')
            filter_list = tf.unstack(padded_filters,axis=4)
            H,W = highestPrimeFactorization(len(filter_list))
            weight_strips = [tf.concat(filter_list[8*i:8*(i+1)],axis=1) for i in range(W)]
            weight_image = tf.concat(weight_strips,axis=2)
            return weight_image
    
        def getActivationImage(activations):
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
            return activation_image
        #######################################################
        ##### PRE-PROCESSING AND DATASET EXTRACTION TOOLS #####
        #######################################################
        def extract_annotations(filename, imgID, coco=coco):
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
            inbounds = tf.less(pts,tf.constant(D,tf.float32))
            inbounds = tf.multiply(tf.to_int32(inbounds), tf.to_int32(tf.greater(pts,0)))
            pts = tf.multiply(pts,tf.to_float(inbounds))
            pts = tf.transpose(pts,[1,0])
            labels = tf.transpose(labels,[1,0])

            padded_image = tf.image.pad_to_bounding_box(cropped_image,tf.to_int32(dX[1]),tf.to_int32(dX[0]),
                                                        tf.to_int32(sideLength),tf.to_int32(sideLength))
            padded_mask = tf.image.pad_to_bounding_box(cropped_mask,tf.to_int32(dX[1]),tf.to_int32(dX[0]),
                                                        tf.to_int32(sideLength),tf.to_int32(sideLength))

            resized_image = tf.image.resize_images(padded_image,tf.constant([D,D]),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # resized_image = resized_image - VGG_MEAN
            resized_mask = tf.image.resize_images(padded_mask,tf.constant([D,D]),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return resized_image, resized_mask, pts, labels

        def scaleDownMaskAndKeypoints(image, mask, pts, labels, D=D):
            mask = tf.image.resize_images(mask,tf.constant([D,D]),tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            pts = tf.multiply(pts,tf.constant(0.5))
            return image, mask, pts, labels
        
        def generate_keypoint_masks(image, mask, keypoints, labels, D=D, L=L):
            X, Y = tf.meshgrid(tf.linspace(0.0,D,D),tf.linspace(0.0,D,D))
            X = tf.reshape(X,[D,D,1])
            Y = tf.reshape(Y,[D,D,1])
            X_stack = tf.tile(X,tf.constant([1,1,17],dtype=tf.int32))
            Y_stack = tf.tile(Y,tf.constant([1,1,17],dtype=tf.int32))

            pts = tf.reshape(keypoints,[1,2,17])
            ptsX, ptsY = tf.split(pts,[1,1],axis=1)
            d1 = tf.square(tf.subtract(X_stack,ptsX))
            d2 = tf.square(tf.subtract(Y_stack,ptsY))

            pt_masks = tf.multiply(tf.divide(tf.constant(1.0),tf.add(d1,d2)+L),L)
            return image, mask, pt_masks, pts, labels
        
        
        #######################################################
        ################### PREPARE DATASET ###################
        #######################################################
        with tf.variable_scope('DataSet'):
            # Initialize train_dataset
            filenames = tf.constant(['{}/COCO_train2014_{:0>12}.jpg'.format(train_img_path,imgID) for imgID in imgIds])
            imgID_tensor = tf.constant(imgIds)
            train_dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames,imgID_tensor))
            # Extract Annotations via coco interface
            train_dataset = train_dataset.map(lambda filename, imgID: tf.py_func(extract_annotations, [filename, imgID], 
                                                                        [filename.dtype, tf.int64, tf.int64, tf.uint8]))
            # All other preprocessing in tensorflow
            train_dataset = train_dataset.map(preprocess_image_tf)
            train_dataset = train_dataset.map(generate_keypoint_masks)
            # BATCH
            train_dataset = train_dataset.shuffle(buffer_size=10000)
            train_dataset = train_dataset.batch(10) # must resize images to make them match
            iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
            images, masks, kpt_masks, pts, labels = iterator.get_next()
            train_init_op = iterator.make_initializer(train_dataset)
        
        #######################################################
        ##################### BUILD GRAPH #####################
        #######################################################

        is_training = tf.placeholder(tf.bool)

        # --------------------------------------------------- #
        # ------------- Resnet V2 50 "Backbone" ------------- #
        # --------------------------------------------------- #
        
        resnet_v2 = tf.contrib.slim.nets.resnet_v2
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, endpoints = resnet_v2.resnet_v2_50(
                inputs=images,
                num_classes=10,
                is_training=is_training,
                reuse=None,
                output_stride=16,
                scope='resnet_v2_50'
                )
    
        model_path = 'checkpoints/resnet_v2_50.ckpt'
        assert(os.path.isfile(model_path))
        # Backbone Variables - remember to exclude all variables above backbone (including block4 and logits)
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['resnet_v2_50/postnorm','resnet_v2_50/logits'])
        # Head variables
        # Note: We would need another set of variables and another initializer to capture the logits as well
        other_variables = tf.contrib.framework.get_variables('resnet_v2_50/postnorm')

        # --------------------------------------------------- #
        # --------------- "Head" Architecture --------------- #
        # --------------------------------------------------- #

        # with tf.variable_scope(HEAD_SCOPE):
        #     # extract attachment tensors from ResNet graph
        #     block1 = endpoints['resnet_v2_50/block1'] # (N,29,29,256)
        #     block2 = endpoints['resnet_v2_50/block2'] # (N,15,15,512)
        #     block3 = endpoints['resnet_v2_50/block3'] # (N,15,15,1024)
        #     block4 = endpoints['resnet_v2_50/block4'] # (N,15,15,2048)

        #     with slim.arg_scope(kernel_size=(3,3), strides=(1,1), activation=tf.nn.relu):
        #         # 1x1 or 3x3 convolutions to reduce dimensionality
        #         block1 = tf.layers.conv2d(block1, 64, kernel_size=(3,3), strides=(1,1))
        #         block2 = tf.layers.conv2d(block2, 128, kernel_size=(3,3), strides=(1,1))
        #         block3 = tf.layers.conv2d(block3, 128, kernel_size=(1,1), strides=(1,1))
        #         block4 = tf.layers.conv2d(block4, 128, kernel_size=(1,1), strides=(1,1))


        




        
        #######################################################
        ############## INITIALIZE AND RUN GRAPH ###############
        #######################################################

        # Call to load pretrained weights for backbone variables
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
        # Call to initialize Head Variables from scratch
        rand_init = tf.variables_initializer(other_variables)

        ###################### SUMMARIES ######################
        filters = tf.contrib.framework.get_variables('resnet_v2_50/conv1/weights')
        weight_image = getFilterImage(filters)
        weights1_summary = tf.summary.image("Layer1_Weights", weight_image)
        # Inspect Layer activations
        block1_activations = getActivationImage(endpoints['resnet_v2_50/block1'])
        block2_activations = getActivationImage(endpoints['resnet_v2_50/block2'])
        block3_activations = getActivationImage(endpoints['resnet_v2_50/block3'])
        block4_activations = getActivationImage(endpoints['resnet_v2_50/block4'])
        block1_unit1_activations = getActivationImage(endpoints['resnet_v2_50/block1/unit_1/bottleneck_v2'])

        block1_summary = tf.summary.image('Block1',block1_activations)
        block2_summary = tf.summary.image('Block2',block2_activations)
        block3_summary = tf.summary.image('Block3',block3_activations)
        block4_summary = tf.summary.image('Block4',block4_activations)
        block1_unit1_summary = tf.summary.image('Block1_Unit1', block1_unit1_activations)
        # Merge all summaries together
        image_summaries = tf.summary.merge(
            [weights1_summary, block1_summary, block1_unit1_summary, block2_summary, block3_summary, block4_summary],
            collections=None, name="image_summaries")


        # Finalize default graph - THIS SEEMS TO PREVENT ADDING A FILEWRITER LATER
        tf.get_default_graph().finalize()

        # Train!
        with tf.Session(graph=graph) as sess:
            # file writer to save graph for Tensorboard
            file_writer = tf.summary.FileWriter('/tmp/KyleNet/1')
            file_writer.add_graph(sess.graph)
            # initialize pretrained variables
            init_fn(sess)
            # initialize other variables
            sess.run(rand_init)
            sess.run(train_init_op) # initialize dataset iterator

            visualization_summaries = sess.run(image_summaries,{is_training: False})
            file_writer.add_summary(visualization_summaries, global_step=1)

            print("Finished")
            return



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)