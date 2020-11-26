import PIL.Image as Img
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
            output = cv2.imread(img)
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

    def _random_select(self, label):
        """
        randomly select a item from a label, this selection is non-replaced
        once the image in the list is exhausted, the selection will be re-filled
        arg:
          label: str, a label of some vignette class
        return:
          a randomly selected element from class {label}
        """
        # interpret the label into the numerical value
        _label = self.label_dict[label]-1
        # find the used label flag in this class
        if self._image_list_used[_label].shape[0] == 0:
            return 0 # the 0 is processed by the augmentation.augment
        # possible index, this logic is somehow tricky...
        _valid_index = np.flatnonzero(self._image_list_used[_label] == 0)
        # selection
        _selection = _valid_index[np.random.randint(_valid_index.shape[0])]
        # upload  _selection to used list
        self._image_list_used[_label][_selection] = 1
        # pick the image selected
        _selected_images = self.image_list[_label][_selection]
        # if all the used_list is exhausted, reset the used list
        if np.all(self._image_list_used[_label]):
            self._image_list_used[_label] = np.zeros_like(self._image_list_used[_label])
        return _selected_images

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

    def _generate_new_color(self, exist_list):
        """
        generate random RGB color not in exist_list
        args:
            exist_list: list of np.ndarray existed colors
        return:
            np.ndarray of shape (n,3), each row as a color is unique
            Note: this color may have a theoretical maximum
                  but I don't think the limit will be reached in my scenario
        """
        while(True):
            color = np.random.choice(a = 256, size = (3), replace = True)
            if not any(np.array_equal(color, x) for x in exist_list):
                exist_list.append(color)
                break
        return color, exist_list

    def _visualize_result(self, collage, mask, dictionary):
        """
        A visualization of result generated

        Discrete legend part credit to
        https://stackoverflow.com/questions/40662475/matplot-imshow-add-label-to-each-color-and-put-them-in-legend/40666123#40666123

        input:
          collage, mask, dictionary, the output of collage generator's .generate() function
        """
        _f, _axarr = plt.subplots(1,2)
        _axarr[0].set_axis_off()
        _im1 = _axarr[0].imshow(collage)
        _axarr[1].set_axis_off()
        _mask = mask if mask.ndim == 2 else mask[:,:,0]
        _values = np.unique(_mask.ravel())
        _im2 = _axarr[1].imshow(_mask)
        # get the colors of the values, according to the colormap used by imshow
        _colors = [_im2.cmap(_im2.norm(value)) for value in _values]
        # create a patch (proxy artist) for every color
        _labels = ["background"] + list([i for i in dictionary if dictionary[i] in _values])
        _patches = [mpatches.Patch(color=_colors[i], label=_labels[i]) for i in range(len(_values))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.show()
        print(dictionary)
