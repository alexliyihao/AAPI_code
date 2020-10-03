import numpy as np
import PIL.Image as Img

def extract_image(filename: str):
    """
    extract the image from the mask
    for colab using, the input is designed for situation where default path have
    both "SegmentationObject" and "JPEGImages" folders
    input:
      filename: str, the name under folder "SegmentationObject"(.png name)
    return:
      np.ndarray, the image extracted by the mask
    """
    assert filename[-3:] == "png"
    pngname = filename
    jpgname = filename[:-3] + "jpg"
    mask_read_in = np.array(Img.open(os.path.join("SegmentationObject",pngname)).convert("RGB"))
    original_image_read_in = np.array(Img.open(os.path.join("JPEGImages",jpgname)).convert("RGB"))
    mask_convert = np.any(mask_read_in, axis = 2)
    original_image_read_in[mask_convert == 0] = 0
    return original_image_read_in

def crop_image(img):
    """
    crop all the 0 paddings aside
    input:
      img: np.ndarray, image to be cropped
    return:
      img: np.ndarray, image cropped
    """
    x = np.nonzero(np.any(img, axis = (0,2)))[0]
    y = np.nonzero(np.any(img, axis = (1,2)))[0]
    xs,xf = x[0],x[-1]
    ys,yf = y[0],y[-1]
    img = img[ys:yf,xs:xf]
    return img
