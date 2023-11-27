
# Remember to init the camera and use cam.BeginAcquisition() and cam.EndAcquisition()
#   before and after the capture function, if these are inside the function everything slows down very much
import os
import PySpin
import numpy as np
import cv2

import time

def blackflyInit():
    print("################ Camera Init ################")
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    print(system)
    print(type(system))
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)
    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print("No camera found")

    cam = cam_list.GetByIndex(0)
    cam.Init()
    ### This part is quite unclear...
    nodemap = cam.GetNodeMap()
    node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
    node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
    acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
    #cam.AcquisitionFrameRate.SetValue(Capture_FPS)
    ##
    node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
    print('Acquisition mode set to continuous, INIT ok')
    print("################################")
    return cam, cam_list, num_cameras, system


def blackflyCapture(cam):
    #cam.BeginAcquisition()
    try:
        time.sleep(0.005)
        image_result = cam.GetNextImage()

        if image_result.IsIncomplete():
            print("not ready")
        else:

            row_bytes = float(len(image_result.GetData()))/float(image_result.GetWidth())
            rawFrame = np.array(image_result.GetData(), dtype="uint8").reshape( (image_result.GetHeight(), image_result.GetWidth()) )
            frame = cv2.cvtColor(rawFrame, cv2.COLOR_BAYER_BG2BGR)
            #frame = cv2.resize(frame,(800,600))

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
    image_result.Release()
    #cam.EndAcquisition()
    return frame


############# NOTES ##################
'''
In case you have problems with re init with the camera, consider the following to clear and release
possible instances made before.... These are implemented in the init but at some point we had problems...

cam.DeInit()
del cam
cam_list.Clear()
time.sleep(1)
system.ReleaseInstance()
'''
