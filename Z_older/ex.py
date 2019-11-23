"""this is my docstring"""

import time

import cv2

CAP = cv2.VideoCapture(0)

ct = 0
while True:
    # Capture frame-by-frame
    ret, frame = CAP.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame',gray)
    title = "Index: " + str(ct)
    cv2.imshow("Index: ", frame)
    ct += 1
    time.sleep(3.0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
CAP.release()
cv2.destroyAllWindows()
