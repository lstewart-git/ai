
# library for handling webcam control in ai applications
import cv2

class cam_engine(object):
    def __init__(self):
        # get webcam handle, set resolution
        self.camera = cv2.VideoCapture(0)

        # set resolution
        ##camera.set(3, 1920)
        #camera.set(4, 1080)
        self.camera.set(3, 1280)
        self.camera.set(4, 720)

    def get_image(self):
        #get an image for analysis
        filename = 'capture.jpg'
        while True:
            return_value,image = self.camera.read()
            cv2.imshow('Press "s" to capture image',image)
            if cv2.waitKey(1)& 0xFF == ord('s'):
                cv2.imwrite(filename,image)
                break

        # run this once loop broken    
        self.camera.release()
        cv2.destroyAllWindows()
        return image

# MAIN PROGRAM ##################################################
if __name__ == "__main__":
    cam_obj = cam_engine()
    cam_obj.get_image()

