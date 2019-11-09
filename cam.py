

import cv2
camera = cv2.VideoCapture(0)
# set resolution
camera.set(3, 1920)
camera.set(4, 1080)

print('\nThis program will capture a frame from the webcam\n')
print('press the "s" key to capture and quit\n')
crpv = input("<ENTER> to start...")

im_ct = 0
while True:
    return_value,image = camera.read()

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    cv2.imshow('Press "s" to save',image)
    filename = 'doggers/' + str(im_ct) + 'cardogger.jpg'
    if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite(filename,image)
        im_ct += 1
    if cv2.waitKey(1)& 0xFF == ord('q'):
       break
# run this once loop broken    
camera.release()
cv2.destroyAllWindows()

