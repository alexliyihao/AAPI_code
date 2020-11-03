import PIL.Image as Img
import numpy as np
from tqdm.notebook import tqdm
from PIL import ImageFilter
import tables
import time

"""
all the insert/append function for collage generator
"""

class _insertion():
    def _canvas_append(self,
                       canvas: np.ndarray,
                       add_point: np.ndarray,
                       img: np.ndarray,
                       mask: np.ndarray = None,
                       mask_label: int = None,
                       mode = "label"):
        """
        the actual working part, add a image to a canvas
        args:
            canvas: np.ndarray, 3-channel canvas
            add_point: tuple of int, the topleft point of the image to be added
            img: np.ndarray, 3-channel, the vignette to be added
            mask: np.ndarray(if it's there), 1-channel, the mask with the canvas
            mask_label: int or 2d np.ndarray , the value of this label onto the mask
            mode: str, "label" or "pattern", how the mask be overwritten,
                if "label", it will use the int mask_label
                if "pattern", it will copy the np.ndarray passed to mask_label pixel-by-pixel
        return:
            canvas: np.ndarray of 3 channels, the canvas with img added.
            mask: np.ndarray of 1 channel, the mask with img's label added.
        """
        assert mode in ["label", "pattern"]
        if mode == "label":
            # if there's no mask
            if type(mask) != np.ndarray:
                # add img to canvas by pixel, only be used in single image preview
                for x in np.arange(img.shape[0]):
                    for y in np.arange(img.shape[1]):
                        if np.any(img[x,y]):
                            canvas[add_point[0]+x, add_point[1]+y] = img[x,y]
                # return canvas
                return canvas
                # if there's a mask
            else:
                # add image to canvas, add label to mask by pixel, return both
                for x in np.arange(img.shape[0]):
                    for y in np.arange(img.shape[1]):
                        if np.any(img[x,y]):
                            canvas[add_point[0]+x, add_point[1]+y] = img[x,y]
                            mask[add_point[0]+x, add_point[1]+y] = mask_label
                return canvas, mask
        else:
            for x in np.arange(img.shape[0]):
                for y in np.arange(img.shape[1]):
                    if np.any(img[x,y]):
                        canvas[add_point[0]+x, add_point[1]+y] = img[x,y]
                        mask[add_point[0]+x, add_point[1]+y] = mask_label[x,y]
            return canvas, mask

    def _init_insert(self,
                     img: np.ndarray,
                     canvas: np.ndarray,
                     mask: np.ndarray,
                     label: int,
                     mode = "pattern"):
        """
        find a random legal position in canvas, append img to canvas and mask
        args:
            img: np.ndarray of 3 channels, the vignette to be added
            canvas: np.ndarray of 3 channels, the canvas
            mask: 2d np.ndarray, the mask
            label: the label to be added
            mode: str, "label" or "pattern", see mode in _canvas_append()
        return:
            canvas: np.ndarray of 3 channels, the canvas with img added.
            mask: np.ndarray of 1 channel, the mask with img's label added.
        """
        _outer_bound = (canvas.shape[0] - img.shape[0], canvas.shape[1] - img.shape[1])
        # select an initial add_point
        _add_point = np.array((np.random.randint(low = self.scanning_constant,
                                           high = _outer_bound[0] - self.scanning_constant),
                        np.random.randint(low = self.scanning_constant,
                                          high = _outer_bound[1] - self.scanning_constant)))
        # create a binary mask of the img
        _img_mask = np.any(img, axis = 2)

        # directly use the _add_point
        canvas, mask = self._canvas_append(canvas = canvas,
                              add_point = _add_point,
                              img = img,
                              mask = mask,
                              mask_label = label,
                              mode = mode)
        return canvas, mask

    def _secondary_insert(self,
                          img: np.ndarray,
                          canvas: np.ndarray,
                          mask: np.ndarray,
                          label: int,
                          patience: int,
                          mode = "label"):
        """
        find a random non-overlapping position in canvas, append img to canvas and mask
        args:
            img: np.ndarray of 3 channels, the vignette to be added
            canvas: np.ndarray of 3 channels, the canvas
            mask: 2d np.ndarray, the mask
            label: the label to be added
            patience: the retry time for finding non-overlapping position
            mode: str, "label" or "pattern", see mode in _canvas_append()
        return:
            canvas: np.ndarray of 3 channels, the canvas with img added,
                if the tries in {patience} times succssfully added the img onto canvas,
                otherwise the original canvas is returned
            mask: np.ndarray of 1 channel, the mask with img added,
                if the tries in {patience} times succssfully added the img onto canvas,
                otherwise the original mask if returned
        """
        _outer_bound = (canvas.shape[0] - img.shape[0], canvas.shape[1] - img.shape[1])
        # select an initial add_point
        _add_point = np.array((np.random.randint(low = self.scanning_constant,
                                         high = _outer_bound[0] - self.scanning_constant),
                      np.random.randint(low = self.scanning_constant,
                                        high = _outer_bound[1] - self.scanning_constant)))
        # create a binary mask of the img
        _img_mask = np.any(img, axis = 2)

        for retry in range(patience):
            # for each time make a small move
            _add_point = _add_point + np.random.randint(low = -1*self.scanning_constant,
                                                    high = self.scanning_constant,
                                                    size = 2)
            # make sure the new value is legal
            _add_point = np.clip(a = _add_point,
                            a_min = (0,0),
                            a_max = _outer_bound)

            # check if there's any overlap
            _check_zone = mask[_add_point[0]:_add_point[0]+_img_mask.shape[0],
                          _add_point[1]:_add_point[1]+_img_mask.shape[1]]
            # if so
            if np.any(np.multiply(_check_zone,_img_mask)) == True:
                #retry for a new point
                continue
            # otherwise add the img to canvas and mask and stop retry
            else:
                canvas, mask = self._canvas_append(canvas = canvas,
                                                add_point = _add_point,
                                                img = img,
                                                mask = mask,
                                                mask_label = label,
                                                mode = mode)
                break
        return canvas, mask

    def _try_insert(self,
                    img: np.ndarray,
                    canvas: np.ndarray,
                    mask: np.ndarray,
                    label: int,
                    patience: int,
                    mode = "label"):
        """
        try to insert img into canvas and mask using a escape-overlapping algorithm
        if the initial point is overlapping, try to "escape" the overlapping
        and append at the first position successfuly escape

        if the initial point is not overlappint, try to find a overlapping point
        and append at the last non-overlapping point before this one

        args:
        img: np.ndarray of 3 channels, the vignette to be added
        canvas: np.ndarray of 3 channels, the canvas
        mask: 2d np.ndarray, the mask
        label: the label to be added
        patience: the retry time for finding non-overlapping position
        mode: str, "label" or "pattern", see mode in _canvas_append()
        return:
        canvas: np.ndarray of 3 channels, the canvas with img added,
            if the tries in {patience} times succssfully added the img onto canvas,
            otherwise the original canvas is returned
        mask: np.ndarray of 1 channel, the mask with img added,
            if the tries in {patience} times succssfully added the img onto canvas,
            otherwise the original mask if returned
        """
        _outer_bound = (canvas.shape[0] - img.shape[0], canvas.shape[1] - img.shape[1])
        # select an initial add_point
        _add_point = np.array((np.random.randint(low = self.scanning_constant,
                                       high = _outer_bound[0] - self.scanning_constant),
                    np.random.randint(low = self.scanning_constant,
                                      high = _outer_bound[1] - self.scanning_constant)))
        # create a binary mask of the img
        _img_mask = np.any(img, axis = 2)

        # check if there's any overlap
        _check_zone = mask[_add_point[0]:_add_point[0]+_img_mask.shape[0],
                _add_point[1]:_add_point[1]+_img_mask.shape[1]]
        # if we start with an overlap, we need to escape from overlap, otherwise we need to find a overlap
        _init_overlapped = np.any(np.multiply(_check_zone,_img_mask))
        # if we are in a finding mode and need to record the last add point
        _last_add_point = _add_point

        # in the patience time
        for retry in range(patience):
            # for each time make a small move
            _add_point = _add_point + np.random.randint(low = -1*self.scanning_constant,
                                                  high = self.scanning_constant,
                                                  size = 2)
            # make sure the new value is legal
            _add_point = np.clip(a = _add_point,
                                 a_min = (0,0),
                                 a_max = _outer_bound)
            # check if there's any overlap
            _check_zone = mask[_add_point[0]:_add_point[0]+_img_mask.shape[0],
                        _add_point[1]:_add_point[1]+_img_mask.shape[1]]
            # check if there's overlap
            _overlap = np.any(np.multiply(_check_zone,_img_mask))
            # if we had a overlap in "escaping"
            if (_overlap == True) and (_init_overlapped == True):
                #retry for a new point
                continue
            # if we met the first non-overlap while escaping
            elif (_overlap == False) and (_init_overlapped == True):
                #stop the finding
                canvas, mask = self._canvas_append(canvas = canvas,
                                                add_point = _add_point,
                                                img = img,
                                                mask = mask,
                                                mask_label = label,
                                                mode = mode)
                break
            # if we are finding but not found
            elif (_overlap == False) and (_init_overlapped == False):
                #record last add_point and retry for a new point
                _last_add_point = _add_point
                continue
            # or we are finding a overlap and found it, we need to use the last
            else:
                canvas, mask = self._canvas_append(canvas = canvas,
                                                 add_point = _last_add_point,
                                                 img = img,
                                                 mask = mask,
                                                 mask_label = label,
                                                 mode = mode)
                break
        return canvas, mask
