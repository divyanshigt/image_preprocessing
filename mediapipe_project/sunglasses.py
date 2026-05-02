# import cv2
# import numpy as np
# import mediapipe as mp
# mp_face_mesh=mp.solutions.face_mesh
# mp_drawing=mp.solutions.drawing_utils
# face_mesh=mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5 
# )
# glasses=cv2.imread("sunsb.png",cv2.IMREAD_UNCHANGED) #read glasses image with alpha channel
# hat=cv2.imread("hat.png",cv2.IMREAD_UNCHANGED) #read hat image with alpha channel
# mode=glasses
# cap=cv2.VideoCapture(0)
# while True:
#     flag,frame=cap.read()
#     if not flag:
#         break
#     rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     #process frame to mediapipe model
#     results=face_mesh.process(rgb_frame)
#     #drawing landmarks
#     if results.multi_face_landmarks:
#         #loop through detected faces(only 1 here)
#         for face_landmarks in results.multi_face_landmarks:
#             h,w,_=frame.shape
#             #left eye=33
#             #right eye=263
#             left_eye_point =face_landmarks.landmark[33] #get landmark point 1 (nose tip)
#             right_eye_point =face_landmarks.landmark[263] #get landmark point 2
#             x1,y1=int(left_eye_point.x*w),int(left_eye_point.y*h)
#             x2,y2=int(right_eye_point.x*w),int(right_eye_point.y*h)
#             #calculate glasses size
#             glasses_width=int(np.hypot(x2-x1,y2-y1))+80 #width is 2 times the distance between eyes
#             #maintain aspectratio
#             glasses_height=int(glasses_width*0.4) #height is half of width
#             if mode==glasses:
#                 img_width, img_height=glasses.shape[1],glasses.shape[0]
#                 overlay=glasses
#                 y_offset_factor=0
#             elif mode==hat:
#                 overlay=hat
#                 y_offset_factor=-img_height
#             #resized glasses image
#             resized_glasses=cv2.resize(glasses,( glasses_width,glasses_height))
#             x_center=(x1+x2)//2
#             y_center=(y1+y2)//2+y_offset_factor
#             #top left corner of glasses 
#             x_offset=int(x_center-img_width/2)
#             y_offset=int(y_center-img_height/2)

#             #split channel
#             overlay_img=resized_glasses[:,:,:3]
#             mask=resized_glasses[:,:,3] #alpha channel as mask

            

#             mask_inv = cv2.bitwise_not(mask)
#             roi=frame[y_offset:y_offset+img_height,x_offset:x_offset+img_width] #region of interest where glasses will be placed
#             if(roi.shape[0]==0 or roi.shape[1]==0):
#                 continue 

#             bg = cv2.bitwise_and(roi, roi, mask=mask_inv) #background where glasses will be placed
#             fg=cv2.bitwise_and(overlay_img,overlay_img,mask=mask) #glasses foreground
#             combined=cv2.add(bg,fg) #combine background and foreground
#             frame[y_offset:y_offset+img_height,x_offset:x_offset+img_width]=combined #place combined image back to frame




          
#     cv2.imshow("MediaPipe Face Mesh",frame)
#     key=cv2.waitKey(1)& 0xFF
#     if key==27: #ESC key to exit
#         break
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
face_mesh=mp_face_mesh.FaceMesh(
static_image_mode=False,
max_num_faces=3,
refine_landmarks=True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5 
)
glasses=cv2.imread("sunglasses.webp",cv2.IMREAD_UNCHANGED) #read glasses image with alpha channel
hat=cv2.imread("riya.webp",cv2.IMREAD_UNCHANGED) #read hat image with alpha channel
mode="glasses"
cap=cv2.VideoCapture(0)
while True:
 flag,frame=cap.read()
 if not flag:
  break
 rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
 #process frame to mediapipe model
 results=face_mesh.process(rgb_frame)
 #drawing landmarks
 if results.multi_face_landmarks:
  #loop through detected faces(only 1 here)
  for face_landmarks in results.multi_face_landmarks:
   h,w,_=frame.shape
   #left eye=33
   #right eye=263
   left_eye_point=face_landmarks.landmark[33]
   right_eye_point=face_landmarks.landmark[263]
   x1,y1=int(left_eye_point.x*w),int(left_eye_point.y*h)
   x2,y2=int(right_eye_point.x*w),int(right_eye_point.y*h)
   #calculate glasses size
   face_width=int(np.hypot(x2-x1,y2-y1))
   if mode=="glasses":
    overlay=glasses
    overlay_width=face_width+50
    overlay_height=int(overlay_width*0.4)
    y_shift=0
   elif mode=="hat":
    overlay=hat
    overlay_width=face_width+140
    overlay_height=int(overlay_width*0.6)
    y_shift=-int(overlay_height*0.51)
   #resized overlay image
   resized_overlay=cv2.resize(overlay,(overlay_width,overlay_height))
   x_center=(x1+x2)//2
   y_center=(y1+y2)//2+y_shift
   #top left corner of overlay 
   x_offset=int(x_center-overlay_width/2)
   y_offset=int(y_center-overlay_height/2)
   if(x_offset<0 or y_offset<0 or x_offset+overlay_width>w or y_offset+overlay_height>h):
    continue
   #split channel
   overlay_img=resized_overlay[:,:,:3]
   mask=resized_overlay[:,:,3] #alpha channel as mask
   mask_inv=cv2.bitwise_not(mask)
   roi=frame[y_offset:y_offset+overlay_height,x_offset:x_offset+overlay_width] #region of interest where overlay will be placed
   bg=cv2.bitwise_and(roi,roi,mask=mask_inv) #background where overlay will be placed
   fg=cv2.bitwise_and(overlay_img,overlay_img,mask=mask) #foreground
   combined=cv2.add(bg,fg) #combine background and foreground
   frame[y_offset:y_offset+overlay_height,x_offset:x_offset+overlay_width]=combined #place combined image back to frame
 cv2.imshow("MediaPipe Face Mesh",frame)
 key=cv2.waitKey(1)&0xFF
 if key==ord('g'):
  mode="glasses"
 elif key==ord('h'):
  mode="hat"
 elif key==27: #ESC key to exit
  break
cap.release()
cv2.destroyAllWindows()