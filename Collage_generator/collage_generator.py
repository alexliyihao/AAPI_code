import PIL.Image as Img
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm
from scipy.stats import mode
from typing import List, Dict, Tuple


class collage_generator(object):

    def __init__(self,
                 label_list: List[str] = ["class_one", "class_two"],
                 canvas_size: Tuple[int,int] = (1024, 1024),
                 example_image: str = "",
                 patience: int = 100,
                 gaussian_noise_constant: float = 25.0,
                 scanning_constant: int = 50,
                 imagedatagenerator = None):
        """
        the initiator
        input:
          label_list:
            list of string, the list of label names
          canvas_size:
            tuple of length 2, the 2d size of the collage
          example_image:
            str/np.ndarray/PIL.imageobject, example_image for background
          patience:
            int, the retry time for each insert if overlap
          gaussian_noise_constant:
            float, define the strength of gaussian noise onto the background
          scanning_constant:
            int, the maximum relocation step try if there's overlapping
          image_data_generator:
            keras.image.imagedatagenerator object, if not None, will replace the default one
        """
        assert len(canvas_size) == 2
        self.label_list = label_list
        self.label_dict = dict(zip(label_list, range(1, len(label_list)+1)))
        self.image_list = [[] for i in range(len(label_list))]
        self.canvas_size = canvas_size
        self.patience = patience
        self.gaussian_noise_constant = gaussian_noise_constant
        self.example_img = self.unify_image_format(example_image)
        self.scanning_constant = scanning_constant
        if imagedatagenerator != None:
            self.datagen = imagedatagenerator
        else:
            self.datagen = ImageDataGenerator(rotation_range = 360,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                shear_range = 0.2,
                                                zoom_range = [1.0, 2.0],
                                                fill_mode = "constant",
                                                cval = 0,
                                                horizontal_flip= True,
                                                vertical_flip= True,
                                                data_format = "channels_last",
                                                dtype = int)

    def unify_image_format(self, img, output_format: str = "np"):
        """
        convert any image input into RGB np.ndarray type
        input:
            img:
              string, the path of image
              or
              np.ndarray/PIL.PngImagePlugin.PngImageFile, the image itself

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

    @property
    def patience(self):
        """
        makes the canvas_size can be access from .patience
        """
        return self._patience

    @patience.setter
    def patience(self, patience: int):
        """
        enforce the update of patience some legal value, the update procedure is
        still from collage_generator.patience = n
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

    @property
    def scanning_constant(self):
        """
        makes the canvas_size can be access from .scanning_constant
        """
        return self._scanning_constant

    @scanning_constant.setter
    def scanning_constant(self, scanning_constant: int):
        """
        enforce the update of patience some legal value, the update procedure is
        still from collage_generator.scanning_constant = n
        """
        assert type(scanning_constant) == int
        assert scanning_constant > 0
        self._scanning_constant = scanning_constant

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
          pre_determined_size: bool, if False, will ask the user for a correct shape with preview
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
        print(f'image succesfully added to label "{label}" with size {size}')

    def single_image_preview(self, img: np.ndarray, canvas_size: Tuple[int,int] = (1,1)):
        """
        generate a size preview for single image vs canvas size
        input:
          img: the image to be added
          canvas_size: the size of canvas
        return:
          canvas: a canvas with img on the topleft corner
        """
        canvas = np.full(shape =(*self.canvas_size,3),
                         fill_value = 255,
                         dtype="uint8")
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
      _outer_bound = (canvas.shape[0] - img.shape[0], canvas.shape[1] - img.shape[1])
      # select an initial add_point
      _add_point = np.array((np.random.randint(low = self.scanning_constant,
                                               high = _outer_bound[0] - self.scanning_constant),
                            np.random.randint(low = self.scanning_constant,
                                              high = _outer_bound[1] - self.scanning_constant)))

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
          _check_zone = mask[_add_point[0]:_add_point[0]+_img_mask.shape[0], _add_point[1]:_add_point[1]+_img_mask.shape[1]]
          # if so
          if np.any(np.multiply(_check_zone,_img_mask)) == True:
            #retry for a new point
            continue
          # otherwise add the img to canvas and mask and stop retry
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
        Just a wrapper, I don't even know np and scipy's multinomial sampling is that weird...
        input:
          prob_distribution: any iterable, the multinomial distribution probability
        output:
          a sample from probability provided
        """
        return np.nonzero(np.random.multinomial(1,prob_distribution))[0][0]


    def generate(self,
                 item_num: int = 200,
                 ratio_list: List[float] = [1.0,0.0],
                 background_color = False
                 ):
        """
        the method to generate a new 3-channel collage and a mask
        input:
          item_num: int, the total number of items in this image
          ratio_list: list[float], the ratio of each cass, the number must be same to the labels
          background_color: bool, if True, will add a background color based on self.example_image

        return:
          _canvas: np.ndarray, self._canvas_size shape, 3 channel, the canvas with images added
          _mask, np.ndarray, self._canvas_size shape, 1 channel, the mask of the canvas
          label_dict: dict, the dictionary of label name and label value
        """

        assert len(ratio_list) == len(self.label_dict)
        # in case the ratio is some weird value
        _ratio_list = np.array(ratio_list)
        _ratio_list = _ratio_list / np.sum(_ratio_list)
        # generate a larger canvas and a larger mask
        _canvas = np.full(shape = (self._canvas_size[0]*2, self._canvas_size[0]*2,3),
                          fill_value = 255,
                          dtype="uint8")
        _mask = np.zeros((self._canvas_size[0]*2, self._canvas_size[0]*2), dtype = "uint8")

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
              _img_transformed = self.datagen.random_transform(_img)
              # append it to the canvas
              _canvas, _mask = self.try_insert(img = _img_transformed,
                                               canvas = _canvas,
                                               mask = _mask,
                                               label = self.label_dict[self.label_list[_class_add]],
                                               patience = self.patience)

        if (background_color):
            # get a background color by extracting the mode
            _color = (mode(self.example_img, axis = 0)[0][0][0] + (np.random.randn(3)*5)).astype(int)
        else:
            # generate a white one
            _color = np.array([255,255,255], dtype = "uint8")
        # create a color backgound of this color
        _color_background = np.multiply(np.ones(shape = _canvas.shape), _color).astype(int)
        # apply the background color to image
        _canvas[_mask == 0] = _color_background[_mask == 0]

        # add a gaussian noise to non-white area
        if (background_color):
            _canvas = _canvas + (np.random.randn(*(_canvas.shape))*self.gaussian_noise_constant)
        else:
            _canvas[_mask != 0] = _canvas[_mask != 0] + np.random.randn(*(_canvas[_mask != 0].shape))*self.gaussian_noise_constant

        _cutbound_x = (int(self.canvas_size[0]/2),int(self.canvas_size[0]/2)+self.canvas_size[0])
        _cutbound_y = (int(self.canvas_size[1]/2),int(self.canvas_size[1]/2)+self.canvas_size[1])
        _cut_canvas = _canvas[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]]
        #make sure the output value is legal
        _cut_canvas = np.clip(_cut_canvas, a_min = 0, a_max = 255).astype(int)
        _cut_mask = _mask[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]]

        return _cut_canvas, _cut_mask, self.label_dict
