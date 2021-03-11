import cv2 as cv
import numpy as np

_CLAHE_CLIP_LIMIT = 3.0
_CLAHE_TILE_GRID_SIZE = (8,8)

def crop_image(image: np.ndarray, width: int = None, height: int = None, ratio: float = None) -> np.ndarray:
    im_height, im_width, _ = image.shape

    if not ratio:
        lower_heigh = (im_height//2) - (height//2)
        upper_heigh = (im_height//2) + (height//2)
        lower_width = (im_width//2) - (width//2)
        upper_width = (im_width//2 + width//2)

        if lower_heigh < 0:
            lower_heigh = 0

        if upper_heigh > im_height:
            upper_heigh = im_height

        if lower_width < 0:
            lower_width = 0
        
        if upper_width > im_width:
            upper_width = im_width
    else:
        lower_heigh = int(im_height*ratio)
        upper_heigh = im_height - int(im_height*ratio)
        lower_width = int(im_width*ratio)
        upper_width = im_width - int(im_width*ratio)

    croped = image[
        lower_heigh:upper_heigh,
        lower_width:upper_width]

    return croped

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv.resize(image, (width, height))

    return resized

def apply_CLAHE(image,
                clipLimit = _CLAHE_CLIP_LIMIT,
                tileGridSize = _CLAHE_TILE_GRID_SIZE):

    #-----Converting image to LAB Color model----------------------------------- 
    labImage = cv.cvtColor(image, cv.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    lChannel,aChannel,bChannel = cv.split(labImage)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    claheLChannel = clahe.apply(lChannel)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    claheLABImage = cv.merge((claheLChannel,aChannel,bChannel))

    #-----Converting image from LAB Color model to RGB model--------------------
    claheFinalImage = cv.cvtColor(claheLABImage, cv.COLOR_LAB2BGR)

    return claheFinalImage