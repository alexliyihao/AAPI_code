import PIL.Image as Img
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm
from scipy.stats import mode
from typing import List, Dict, Tuple

class collage_generator(object):

    def __init__(self,
                 label_list: List[str] = ["one", "two"],
                 canvas_size: Tuple[int,int] = (1024, 1024),
                 example_image = str,
                 patience: int = 10,
                 gaussian_noise_constant: float = 100.0,
                 imagedatagenerator = None):
        """
        the initiator
        input:
          label_list: list of string, the list of label names
          canvas_size: tuple of length 2, the 2d size of the collage
          example_image: str/np.ndarray/PIL.imageobject, example_image for background
          patience: int, the retry time for each insert if overlap
          gaussian_noise_constant: float, define the strength of gaussian noise onto the background
        """
        assert len(canvas_size) == 2
        self.label_list = label_list
        self.label_dict = dict(zip(label_list, range(1, len(label_list)+1)))
        self.image_list = [[] for i in range(len(label_list))]
        self.canvas_size = canvas_size
        self.patience = patience
        self.gaussian_noise_constant = gaussian_noise_constant
        self.example_img = self.unify_image_format(example_image)
        if imagedatagenerator != None:
            self.datagen = imagedatagenerator
        else:
            self.datagen = ImageDataGenerator(rotation_range = 360,
                                              width_shift_range=0.1,
                                              height_shift_range=0.1,
                                              shear_range = 0.2,
                                              zoom_range = 0.2,
                                              fill_mode = "constant",
                                              cval = 0,
                                              horizontal_flip= True,
                                              vertical_flip= True,
                                              data_format = "channels_last",
                                              dtype = int)

    def unify_image_format(self, img, output_format: str = "np"):
        """
        convert any input image into RGB np.ndarray type
        input:
            img: string/np.ndarray/PIL.PngImagePlugin.PngImageFile, the path of image or image itself
            output_format: "np" or "PIL"
        return:
            output: RGB np.ndarray, the image
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

    @property
    def patience(self):
        """
        makes the canvas_size can be access from .canvas_size
        """
        return self._patience

    @patience.setter
    def patience(self, patience: int):
        """
        enforce the update of patience some legal value, the update still from
        collage_generator.patience = n
        """
        assert type(patience) == int
        assert patience > 0
        self._patience = patience

    @property
    def canvas_size(self):
        """
        makes the canvas_size can be access from .canvas_size
        """
        return self._canvas_size

    @canvas_size.setter
    def canvas_size(self, canvas_size: Tuple[int,int]):
        """
        enforce the update of canvas size some legal value, the update still from
        collage_generator.canvas_size = (x,y)
        """
        assert len(canvas_size) == 2
        assert canvas_size[0] > 0
        assert canvas_size[1] > 0
        self._canvas_size = (canvas_size[1], canvas_size[0])

    def add_label(self, new_label: str):
        """
        add a new label into the generator
        input:
          new_label: string, the name of this label
        """
        #add new label into the list
        self.label_list.append(new_label)
        #meanwhile update the label_dict and the image storage
        self.label_dict = dict(zip(self.label_list, range(1, len(self.label_list)+1)))
        self.image_list.append([])

    def add_image(self,
                  img,
                  label: str,
                  size: Tuple[int,int] = (80,60),
                  pre_determined_size: bool = False):
        """
        add an image into the image storages with specified label and size
        input:
          img: str/np.ndarray/PIL.PngImagePlugin.PngImageFile, the path of the image or the image object
          label: str, the label of the image, which must be in the label list
          size: tuple of int, the size of the image in the image storage
        """
        assert label in self.label_list
        read_in = self.unify_image_format(img, output_format = "PIL")

        # confirm the correct size of the image
        if pre_determined_size == True:
            np_img = np.array(read_in.resize(size))
        else:
            size_okay = False
            while (not size_okay):
                # resize it and preview
                np_img = np.array(read_in.resize(size))
                plt.imshow(self.single_image_preview(np_img,
                                                    canvas_size = self.canvas_size))
                plt.show()
                # check if original size okay
                size_okay_input = input("is the size okay? [y/n] ")

                # if the size if acceptable, leave the loop
                if size_okay_input == "y" or size_okay_input == "Y":
                    size_okay = True

                # otherwise update a new shape
                else:
                    print(f"current size is {size}, please specify a new one")
                    input_ok = False
                    while not input_ok:
                      new_x = input("new x: ")
                      new_y = input("new y: ")
                      try:
                        new_x = int(new_x)
                        new_y = int(new_y)
                        input_ok = True
                      except:
                        print("invalid input")
                    size = (new_x, new_y)

        self.image_list[self.label_dict[label]-1].append(np_img)
        print(f'image succesfully added to label "{label}"')

    def single_image_preview(self, img: np.ndarray, canvas_size: Tuple[int,int] = (1,1)):
        """
        generate a size preview for single image vs canvas size
        input:
          img: the image to be added
          canvas_size: the size of canvas
        return:
          canvas: a canvas with img on the topleft corner
        """
        canvas = np.ones((*self.canvas_size,3), dtype="uint8")*255
        canvas = self.canvas_append(canvas = canvas,
                                    add_point = (0,0),
                                    img = img)
        return canvas

    def canvas_append(self,
                      canvas: np.ndarray,
                      add_point: np.ndarray,
                      img: np.ndarray,
                      mask: np.ndarray = None,
                      mask_label: int = None):
        """
        the actual working part, add a image to a canvas
        input:
          canvas: np.ndarray, 3-channel canvas
          add_point: tuple of int, the topleft point of the image to be added
          img: np.ndarray, 3-channel, the image to be added
          mask: np.ndarray(if it's there), 1-channel, the mask with the canvas
          mask_label: int(if mask if np.ndarray), the value of this label onto the mask
        """
        # if there's no mask
        if type(mask) != np.ndarray:
          # add img to canvas by pixel, only be used in single image preview
          for x in np.arange(img.shape[0]):
            for y in np.arange(img.shape[1]):
              if not np.any(img[x,y]):
                pass
              else:
                for j in np.arange(3):
                  canvas[add_point[0]+x, add_point[1]+y,j] = img[x,y,j]
          # return canvas
          return canvas
        # if there's a mask
        else:
          # add image to canvas, add label to mask by pixel, return both
          for x in np.arange(img.shape[0]):
            for y in np.arange(img.shape[1]):
              if not np.any(img[x,y]):
                pass
              else:
                for j in np.arange(3):
                  canvas[add_point[0]+x, add_point[1]+y,j] = img[x,y,j]
                mask[add_point[0]+x, add_point[1]+y] = mask_label
          return canvas, mask

    def try_insert(self,
                   img: np.ndarray,
                   canvas: np.ndarray,
                   mask: np.ndarray,
                   label: int,
                   patience: int):
      """
      try to insert img into canvas and mask, if there's any overlap, redo it, retry for patience times
      input:
        img: np.ndarray of 3 channels, the image of item to be added
        canvas: np.ndarray of 3 channels, the canvas
        mask: np.ndarray of 1 channel, the mask
        label: int, the value of the label onto the mask
        patience: int, the number of tries adding img to canvas and mask
      return:
        canvas: np.ndarray of 3 channels, the canvas with img added,
                if the tries in {patience} times succssfully added the img onto canvas,
                otherwise the original canvas is returned
        mask: np.ndarray of 1 channel, the mask with img added,
                if the tries in {patience} times succssfully added the img onto canvas,
                otherwise the original mask if returned
      """
      for retry in range(patience):
          # select a add_point
          _add_point = np.array((np.random.randint(0, canvas.shape[0] - img.shape[0]),
                                np.random.randint(0, canvas.shape[1] - img.shape[1])))
          _img_mask = np.any(img, axis = 2)
          _check_zone = mask[_add_point[0]:_add_point[0]+_img_mask.shape[0], _add_point[1]:_add_point[1]+_img_mask.shape[1]]
          # check if there's any overlap
          # if so
          if np.any(np.multiply(_check_zone,_img_mask)) == True:
            #retry for a new point
            continue
          # otherwise add the img to canvas and mask
          else:
            canvas, mask = self.canvas_append(canvas = canvas,
                                              add_point = _add_point,
                                              img = img,
                                              mask = mask,
                                              mask_label = label)
            break
      return canvas, mask

    def multinomial_distribution(self, prob_distribution):
        """
        I don't even know np and scipy's multinomial sampling is that weird...
        input:
          prob_distribution: any iterable, the multinomial distribution probability
        output:
          a single sample from this probability
        """
        return np.nonzero(np.random.multinomial(1,prob_distribution))[0][0]


    def generate(self,
                 item_num: int = 10,
                 ratio_list: List[float] = [1.0,0.0]
                 ):
        """
        the method to generate a new 3-channel collage and a mask
        input:
          item_num: int, the total number of items in this image
          ratio_list: list[float], the ratio of each cass, the number must be same to the labels

        return:
          _canvas: np.ndarray, self._canvas_size shape, 3 channel, the canvas with images added
          _mask, np.ndarray, self._canvas_size shape, 1 channel, the mask of the canvas
          label_dict: dict, the dictionary of label name and label value

        """

        assert len(ratio_list) == len(self.label_dict)
        # in case the ratio is some weird value
        _ratio_list = np.array(ratio_list)
        _ratio_list = _ratio_list / np.sum(_ratio_list)
        # generate a new canvas and a new mask
        _canvas = np.ones((*self._canvas_size,3), dtype="uint8")*255
        _mask = np.zeros(self._canvas_size, dtype = "uint8")

        #for each iteration
        for _num_count in tqdm(np.arange(item_num), desc = "Generating...", leave = True):
            # generate a random class from the distribution
            _class_add = self.multinomial_distribution(_ratio_list)
            # find the image_list for this class
            _image_list = self.image_list[_class_add]
            # if the class is empty, skip
            if len(_image_list) == 0:
              pass
            # if it's not empty
            else:
              # choose a random images
              _img = _image_list[np.random.randint(len(_image_list))]
              # randomly transform it
              _img_transformed = self.datagen.random_transform(_img)
              # append it to the canvas
              _canvas, _mask = self.try_insert(img = _img_transformed,
                                               canvas = _canvas,
                                               mask = _mask,
                                               label = self.label_dict[self.label_list[_class_add]],
                                               patience = self.patience)


        # get a background color by extracting the mode
        _color= mode(self.example_img, axis = 0)[0][0][0]
        # create a color backgound of this color
        _color_background = np.multiply(np.ones(shape = (*_mask.shape,3)), _color).astype(int)
        # clip the value between 0 and 255
        _color_background = np.clip(_color_background, a_min = 0, a_max = 255)
        # apply the color to image
        _canvas[_mask == 0] = _color_background[_mask == 0]
        # add a gaussian noise
        _canvas = _canvas + (np.random.randn(*(_canvas.shape))*self.gaussian_noise_constant)
        _canvas = np.clip(_canvas, a_min = 0, a_max = 255).astype(int)

        return _canvas, _mask, self.label_dict
