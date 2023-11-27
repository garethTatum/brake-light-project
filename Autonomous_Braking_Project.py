import os
import glob
import cv2

# Cam_Demo Test
%pip install opencv-python
!python /content/drive/MyDrive/brake-light-project/cam_demo.py --recording 1 --reso 608 --video "C:\Users\gtatu\OneDrive\Documents\InspiritAI\Videos-Unprocessed\d9bd0679-1826b1d8.mov"

# Loop
# fileList = glob.glob("/content/drive/MyDrive/AI-Project/Videos-Unprocessed/*.mov")
fileList = glob.glob("C:\Users\gtatu\OneDrive\Documents\InspiritAI\Videos-Unprocessed\*.mov")

# videoBrake = cv2.VideoWriter("/content/drive/MyDrive/AI-Project/Brake-Out/brake-out.mov", cv2.VideoWriter_fourcc(*'mp4v'), 15, (1280, 720), True)

index = 0

# Training
for vidFile in fileList:
  print("File: ", vidFile) 
  # os.system("python /content/drive/MyDrive/brake-light-project/video_demo.py --cfg /content/drive/MyDrive/brake-light-project/cfg/yolov3.cfg --weights /content/drive/MyDrive/brake-light-project/yolov3.weights --video " + vidFile + " --index " + str(index))
  os.system("python C:\Users\gtatu\OneDrive\Documents\InspiritAI\Code\brake-light-project\cam_demo.py --recording 1 --reso 608 --video "+ vidFile)
  cap = cv2.VideoCapture(vidFile)
  if (cap.isOpened()== False): 
    print("Error opening video stream or file")
  # Read until video is completed
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if ret == True:     
      # Display the resulting frame
      # cv2.imshow('Frame',frame)
     # Press Q on keyboard to  exit
      # if cv2.waitKey(25) & 0xFF == ord('q'):
      #   break
  # Break the loop
  # else: 
    break
    
  # When everything done, release the video capture object
  # newVideo = VideoFileClip("/content/drive/MyDrive/AI-Project/Brake-Out/brake-out.mov")
  # brakeOut = VideoFileClip("/content/drive/MyDrive/AI-Project/Brake-Out/brake-out-test.mov")

  # brakeOut = concatenate_videoclips([newVideo, brakeOut])
  # brakeOut.write_videofile("/content/drive/MyDrive/AI-Project/Brake-Out/brake-out-test.mov")

  index += 1
  cap.release()

