import numpy as np
import cv2


def show_img(img, title='example'):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # to cv2 bgr
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def read_img(path) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def letterbox_resize(img, size, show=False):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw = img.shape[:2]
    h, w = size
    if ih > iw :
        nh = h
        nw = int(h / ih * iw)
    elif ih < iw:
        nw = w
        nh = int(w / iw * ih)
    else:
        nh, nw = h, w
    # resize the image to small side is

    img_resize = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_image = np.empty((size[0], size[1], 3), dtype=np.uint8)
    new_image[...] = (128, 128, 128)
    try:
        if ih < iw:
            new_image[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:] = img_resize
        elif ih > iw:
            new_image[(h - nh) // 2:, (w - nw) // 2:(w - nw) // 2 + nw, :] = img_resize
        else:
            new_image = img_resize


    except Exception as e:
        print(e)
        print("image shape:{}, resize shape: {}".format(img.shape, (nh, nw)))

    if show:
        # print(new_image.shape)
        show_img(new_image)

    return new_image

def image_inference_preprocess(img, reshape_size):
    img = letterbox_resize(img, reshape_size, show=False)
    img = np.array(img, dtype=np.float32)
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img
