from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import argparse
import pickle as pkl

from brake_utils.preprocess_predict import brakecheck4
from brake_utils.tracking import CentroidTracker
from brake_utils.random_forest_manual_train import RandomForest

# from google.colab.patches import cv2_imshow

# def get_test_input(input_dim, CUDA):
#     img = cv2.imread("C:/Users\gtatu\OneDrive\Documents\InspiritAI\Code\brake-light-project\imgs\messi.jpg")
#     img = cv2.resize(img, (input_dim, input_dim))
#     img_ =  img[:,:,::-1].transpose((2,0,1))
#     img_ = img_[np.newaxis,:,:,:]/255.0
#     img_ = torch.from_numpy(img_).float()
#     img_ = Variable(img_)

#     if CUDA:
#         img_ = img_.cuda()

#     return img_

###
# My (Gareth's) Functions
def distance_estimator(boxes):
    distanceEstimates = []

    if len(boxes) > 0:
        for box in boxes:
            boxWidth = abs(box[0]-box[2])

            # Perspective Geometry Formula
            distanceEstimate = (800/boxWidth) * (3/2)
            distanceEstimates.append(distanceEstimate)

        # Average (remove later)
        return np.mean(distanceEstimates)
    else:
        return 0

def velocity_estimator(boxes, previous_boxes):
    box_area_sum = 0
    previous_box_area_sum = 0

    for box in boxes:
        boxArea = abs((box[0]-box[2])*(box[1]-box[3]))
        box_area_sum += boxArea
    
    for box in previous_boxes:
        boxArea = abs((box[0]-box[2])*(box[1]-box[3]))
        previous_box_area_sum += boxArea
    
    return previous_box_area_sum - box_area_sum   
### 

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.
    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    print("writing")
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = [0,0,255]

    if label == "car":
        cv2.rectangle(img, c1, c2,color, 4)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2_class = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2_class,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    parser.add_argument("--recording", dest = "record", help = "Recording = 1, default = 0", default = 0)
    parser.add_argument("--video", dest = "input_video")
    parser.add_argument("--index", dest="index", default="0")
    return parser.parse_args()

# if __name__ == '__main__':
def cam_demo_main(input_video, confidence = 0.25, nms_thresh = 0.4, reso = "160", record = 0, index = "0"):
    print("reso = ", reso)
    cfgfile = "D:/InspiritAI/Code/brake-light-project/cfg/yolov3.cfg"
    weightsfile = "D:/InspiritAI/Code/brake-light-project/yolov3.weights"
    num_classes = 80
    print("nms_thresh = ", nms_thresh)

######################
    # args = arg_parse()
    # print("args = ", args)
    confidence = float(confidence)
    nms_thesh = float(nms_thresh)
    print("nms_thesh = ", nms_thesh)
    record = bool(record)
    start = 0
    CUDA = torch.cuda.is_available()
    ct = CentroidTracker()

# Random forest init & train
    rf = RandomForest()
    rf.train()
######################

### Warning sign for live feed
    ######################
    orig_imgfigure = cv2.imread("D:/InspiritAI/Code/brake-light-project/brake_utils/warning.png",-1)
    orig_mask = orig_imgfigure[:,:,3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)
    orig_imgfigure = orig_imgfigure[:,:,0:3]
    orig_figureHeight, orig_figureWidth = orig_imgfigure.shape[:2]
    figureHeight = 50
    figureWidth = 50
    imgfigure = cv2.resize(orig_imgfigure, (figureWidth, figureHeight), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (figureWidth,figureHeight), interpolation = cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (figureWidth,figureHeight), interpolation = cv2.INTER_AREA)
    ###############

    num_classes = 80
    bbox_attrs = 5 + num_classes
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    # model.net_info["height"] = args.reso
    model.net_info["height"] = reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()
    model.eval()

    # Video source
    # cap = cv2.VideoCapture("/home/henry/Videos/CheckedVideos/fourth_run/march2020/2020-03-02 09:39:34.944830/2020-03-02 09:39:34.944830.avi")
    # cap = cv2.VideoCapture(args.input_video)
    cap = cv2.VideoCapture(input_video)
    print("Capturing ", input_video)
    # In case a recording is needed
    # if record:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("D:/InspiritAI/Break-Out/Cam_Demo_" + index + ".mov", fourcc, 30.0, (800, 600)) # (1280,720))

    assert cap.isOpened(), 'Cannot capture source'

    ###
    # Gareth's Variables
    boxes = []
    previous_boxes = []
    average_areas = []
    average_change_in_areas = []
    number_of_cars_by_frame = []
    brake_lights_on = []
    ###

    frames = 0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = time.time()
    while cap.isOpened():
        # print("Cap opened")
        
        ret, frame = cap.read()
        # print("ret = ", ret, "\nframe = ", frame)
        # print("frame = ", frame)

        if ret:
            # brake_lights_on.append(0)
            frame = cv2.resize(frame, (800,600))
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                # cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Breaking")
                    break
                continue

            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            classes = load_classes('D:/InspiritAI/Code/brake-light-project/data/coco.names')
            colors = pkl.load(open("D:/InspiritAI/Code/brake-light-project/pallete", "rb"))

            #OWN IMPLEMENTATIONS START HERE - Finnish Grad Students
            #########################################
            #removes all but cars in the specified triangular area
            output = list(filter(lambda x: (x[-1] == 2) and x[1] > 200 and x[3]< 600, output))
            #and x[1] > 200 and x[3]< 600
            # Gareth - Assuming that x[1] is upper left x, x[2] is upper left y, etc...
            bounding_boxes = list(map(lambda x: ([x[1].item(), x[2].item(), x[3].item(), x[4].item()]), output))
            boxes = bounding_boxes
            prediction_time = []
            begin = time.time()
            brake_lights = list(map(lambda x: brakecheck4(orig_im[int(x[1]):int(x[3]),int(x[0]):int(x[2])], rf), bounding_boxes))
            print(f"{brake_lights=}")
            # print("Prediction time: {} ms".format((time.time()-begin)*1000))
            prediction_time.append(time.time()-begin)
            # list(map(lambda x: cv2.rectangle(orig_im, tuple(x[1:3].int()), tuple(x[3:5].int()), (0,255,0), 1), output))
            # print("output = ", output)
            list(map(lambda x: cv2.rectangle(orig_im, [int(s) for s in x[1:3]], [int(s) for s in x[3:5]], (0,255,0), 1), output))
            # [*c1.tolist(), *c2.tolist()]

            objects, brake_light_statuses = ct.update(bounding_boxes, brake_lights)
            num_brake_lights = 0
            for (objectID, centroid), brake_light_status in zip(objects.items(), brake_light_statuses.values()):
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                if brake_light_status:
                    num_brake_lights += 1
                    roi = orig_im[centroid[1]:centroid[1]+figureHeight, centroid[0]:centroid[0]+figureWidth]
                    try:
                        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    except Exception as err:
                        print(f"Unexpected {err=}, {type(err)=}, {input_video=}, frame=", frames,"/", num_frames)
                        raise
                    # else:
                    #     print("WTF")

                    roi_fg = cv2.bitwise_and(imgfigure,imgfigure,mask = mask)
                    dst = cv2.add(roi_bg,roi_fg)
                    orig_im[centroid[1]:centroid[1]+figureHeight, centroid[0]:centroid[0]+figureWidth] = dst
                    # print("objectID = ", objectID, "\ncentroid = ", centroid)
            # if num_brake_lights > 0:
            #     brake_lights_on.append(1)
            # elif num_brake_lights == 0:
            #     brake_lights_on.append(0)
            if num_brake_lights > 0:
                brake_lights_on.append(0)
            elif num_brake_lights == 0:
                brake_lights_on.append(1)
            
            cv2.line(orig_im,(200,600),(200,0),(255,0,0),1)
            cv2.line(orig_im,(600,600),(600,0),(255,0,0),1)

            #For saving video
            # if record == "1":
            # print("prediction_time = ", prediction_time)
            # out.write(cv2.flip(orig_im, -1))
            out.write(orig_im)
            #For showing video
            # cv2_imshow(orig_im)

            if (len(previous_boxes) != 0):
                average_change_in_areas.append(velocity_estimator(boxes, previous_boxes))
            else:
                average_change_in_areas.append(0)
                # print("Test")
                # print("Chage in area: ", average_change_in_areas[len(average_change_in_areas)-1])
            previous_boxes = boxes
            number_of_cars_by_frame.append(len(boxes))
            average_areas.append(distance_estimator(boxes))
            # print("# of brake lights: ", brake_lights_on[len(brake_lights_on)-1])
            

            # Need to return average_change_in_areas, number_of_cars_by_frame, average_areas

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("Breaking")
                break
            frames += 1
            # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            print("Not read correctly")
            break
    
    print("Cars by frame = ", number_of_cars_by_frame)
    return [number_of_cars_by_frame, average_areas, average_change_in_areas, brake_lights_on]

