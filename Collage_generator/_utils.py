import PIL.Image as Img
import numpy as np
"""
all the utility functions for the collage_generator class
"""
class _utils():
    def _unify_image_format(self, img, output_format: str = "np"):
        """
        convert any image input into RGB np.ndarray type
        args:
            img:
              string, the path of image
              or
              np.ndarray/PIL.PngImagePlugin.PngImageFile, the image object

            output_format: string, "np" or "PIL"
        return:
            output:
              if output_format = "np", return RGB np.ndarray,
              if output_format = "PIL", return PIL image object
        """
        assert output_format in ["np", "PIL"]
        # if it's a string
        if type(img) == str:
            output = np.array(Img.open(img).convert("RGB"))
        # if it's a np.ndarray
        elif type(img) == np.ndarray:
            # if it's the correct format
            if img.shape[2] == 3:
                output = img
            # if it's RGBA
            else:
                output = np.array(Img.fromarray(img).convert("RGB"))
        # if it's a PIL
        elif type(img) == PIL.PngImagePlugin.PngImageFile:
            output = np.array(extract_image.convert("RGB"))
        else:
            raise TypeError("Invalid image input")

        if output_format == "PIL":
            output = Img.fromarray(output)

        return output

    def _random_select(self, list_x):
        """
        wrapper for randomly selected from a list
        arg:
          list_x, a list
        return:
          a randomly selected element from list_x
        """
        return list_x[np.random.randint(len(list_x))]

    def _multinomial_distribution(self, prob_distribution):
        """
        Just a wrapper, I don't even know np and scipy's multinomial sampling is that weird...
        args:
          prob_distribution: any iterable, the multinomial distribution probability
        output:
          a sample from probability provided
        """
        return np.nonzero(np.random.multinomial(1,prob_distribution))[0][0]

    def _crop_image(self, img):
        """
        crop all the 0 paddings aside
        args:
          img: np.ndarray, image to be cropped
        return:
          img: np.ndarray, image cropped
        """
        if img.ndim != 2:
            x = np.nonzero(np.any(img, axis = (0,2)))[0]
            y = np.nonzero(np.any(img, axis = (1,2)))[0]
        else:
            x = np.nonzero(np.any(img, axis = (0)))[0]
            y = np.nonzero(np.any(img, axis = (1)))[0]
        xs,xf = x[0],x[-1]
        ys,yf = y[0],y[-1]
        img = img[ys:yf,xs:xf]
        return img

    def _cut_circle_edges(self, mask, center = (1, 1), radius = 3, circle_mask_label = 100):
        """
        create a round mask on a 2D mask with label = circle_mask_label,
        used to create a "round distribution" of cluster
        """
        m,n = mask.shape
        a,b = center
        r = radius
        y,x = np.ogrid[-a:m-a, -b:n-b]
        _mask = (x*x + y*y > r*r)
        mask[_mask] = circle_mask_label
        return mask
