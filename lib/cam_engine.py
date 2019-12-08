
# library for handling webcam control in ai applications
import cv2

class cam_engine(object):
    def __init__(self):
        # get webcam handle, set resolution
        self.camera = cv2.VideoCapture(0)
        # camera resolution
        self.cam_width = 1280
        self.cam_height = 720

        # set central focus box
        self.box_size = int(0.7 * self.cam_height)
        cen_x = int(self.cam_width /2)
        cen_y = int(self.cam_height / 2)
        self.ULx = int(cen_x - self.box_size/2)
        self.ULy = int(cen_y + self.box_size/2)
        self.LRx = int(cen_x + self.box_size/2)
        self.LRy = int(cen_y - self.box_size/2)
        self.bound_boxUL = (self.ULx, self.ULy)
        self.bound_boxLR = (self.LRx, self.LRy)
        # set resolution
        self.camera.set(3, self.cam_width)
        self.camera.set(4, self.cam_height)

    def get_image(self):
        #get an image for analysis
        filename = 'capture.jpg'
        filename2 = 'captghure2.jpg'
        while True:
            return_value,image = self.camera.read()
            raw_image = image.copy()
            #cv2.line(image, (50, 50), (150, 150), (0,255,0), 2)
            cv2.rectangle(image, self.bound_boxUL, self.bound_boxLR, (0,255,0), 1) 
            cv2.imshow('Press "spacebar" to capture image',image)
            if cv2.waitKey(1)& 0xFF == ord(' '):
                #cv2.imwrite(filename,raw_image)
                break

        # run this once loop broken    
        self.camera.release()
        cv2.destroyAllWindows()
        # crop the image to the part of interest
        crop_img = raw_image[self.LRy:self.ULy, self.ULx:self.LRx]
        #cv2.imwrite(filename2,crop_img)
        # returned as BGR from opencv need RGB for rest of world
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)        
        return crop_img

# MAIN PROGRAM ##################################################
if __name__ == "__main__":
    cam_obj = cam_engine()
    cam_obj.get_image()

