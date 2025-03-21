import cv2
import urllib.request
import numpy as np

# my_ip = '192.168.1.55' # ip given by ip-webcam

# while True:
#     req = urllib.request.urlopen('http://'+my_ip+':8080/shot.jpg')
#     arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
#     img = cv2.imdecode(arr, -1) # 'Load it as it is'

#     if cv2.waitKey(1) == 27:
#         break
#     cv2.imshow('Its Me', img)

# cv2.destroyAllWindows()



cap = cv2.VideoCapture('http://192.168.1.55:4747/video')

while True:
    ret, frame = cap.read()

    cv2.imshow("frame", frame)
    cv2.waitKey(1)