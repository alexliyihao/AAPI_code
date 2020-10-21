import PIL.Image as Img
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm
from scipy.stats import mode
from typing import List, Dict, Tuple
from PIL import ImageFilter
import tables
import time
import os

class collage_generator(object):

    def __init__(self,
                 label_list: List[str] = ["class_one", "class_two"],
                 canvas_size: Tuple[int,int] = (1024, 1024),
                 cluster_size: Tuple[int,int] = (500,500),
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
        super(collage_generator, self).__init__()
        assert len(canvas_size) == 2
        self.label_list = label_list
        self.label_dict = dict(zip(label_list, range(1, len(label_list)+1)))
        self.image_list = [[] for i in range(len(label_list))]
        self.canvas_size = canvas_size
        self.patience = patience
        self.gaussian_noise_constant = gaussian_noise_constant
        self.example_img = self.unify_image_format(example_image)
        self.scanning_constant = scanning_constant
        self.max_component_size = np.array([0,0])
        self.cluster_size = cluster_size
        if imagedatagenerator != None:
            self.datagen = imagedatagenerator
        else:
            self.datagen = ImageDataGenerator(#rotation_range = 360,
                                              #width_shift_range=0.1,
                                              #height_shift_range=0.1,
                                              #shear_range = 0.1,
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
    def cluster_size(self):
        """
        makes the canvas_size can be access from .canvas_size
        """
        return self._cluster_size

    @cluster_size.setter
    def cluster_size(self, cluster_size: Tuple[int,int]):
        """
        enforce the update of canvas size some legal value, the update still from
        collage_generator.cluster_size = (x,y)
        """
        assert len(cluster_size) == 2
        assert cluster_size[0] > 0
        assert cluster_size[1] > 0
        self._cluster_size = (cluster_size[1], cluster_size[0])

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
                  original_image_size = False,
                  size: Tuple[int,int] = (80,60),
                  pre_determined_size: bool = False):
        """
        add an image into the image storages with specified label and size
        input:
          img: str/np.ndarray/PIL.PngImagePlugin.PngImageFile, the path of the image or the image object
          label: str, the label of the image, which must be in the label list
          size: tuple of int, the size of the image in the image storage
          original_image_size: bool, if true, will use the original image size
          pre_determined_size: bool, if False, will ask the user for a correct shape with preview
        """
        assert label in self.label_list

        if original_image_size == True:
            np_img = self.unify_image_format(img, output_format = "np")
            size = np_img.shape[:2]
        else:
            read_in = self.unify_image_format(img, output_format = "PIL")
            # confirm the correct size of the image
            if pre_determined_size == True:
                np_img = np.array(read_in.resize(size,resample = Img.NEAREST))
            else:
                size_okay = False
                while (not size_okay):
                    # resize it and preview
                    np_img = np.array(read_in.resize(size,resample = Img.NEAREST))
                    plt.imshow(self._single_image_preview(np_img,
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
        size = np.array(size)
        self.image_list[self.label_dict[label]-1].append(np_img)
        self.max_component_size = np.max(np.vstack((self.max_component_size, size)), axis = 0)
        print(f'image succesfully added to label "{label}" with size {size}')

    def import_images_from_directory(self, root_path):
        """
        directly import all the images class under root_path
        arg:
          root_path: the root path loading images
        return:
          image_list: 2D list of RGB np.ndarray, each label will have a individual list
          class_name_list: 1D list of string, number of label
        """
        for label in sorted(os.listdir(root_path)):
            if label == "background.png":
              continue
            self.add_label(new_label = label)
            for img in os.listdir(os.path.join(root_path, label)):
                if label == "arteriole":
                    self.add_image(img = os.path.join(root_path, label, img),
                                   label = label,
                                   original_image_size = False,
                                   size = (80,80),
                                   pre_determined_size = True
                                   )
                elif label == "artery":
                    self.add_image(img = os.path.join(root_path, label, img),
                                   label = label,
                                   original_image_size = False,
                                   size = (300,300),
                                   pre_determined_size = True
                                   )
                elif label == "glomerulus":
                    self.add_image(img = os.path.join(root_path, label, img),
                                   label = label,
                                   original_image_size = False,
                                   size = (300,300),
                                   pre_determined_size = True
                                   )
                elif label == "proximal_tubule":
                    self.add_image(img = os.path.join(root_path, label, img),
                                   label = label,
                                   original_image_size = False,
                                   size = (150,150),
                                   pre_determined_size = True
                                   )
                elif label == "distal_tubule":
                    self.add_image(img = os.path.join(root_path, label, img),
                                   label = label,
                                   original_image_size = False,
                                   size = (200,200),
                                   pre_determined_size = True
                                   )
                #else:
                #    self.add_image(img = os.path.join(root_path, label, img),
                #                  label = label,
                #                  original_image_size = True)

    def _single_image_preview(self, img: np.ndarray, canvas_size: Tuple[int,int] = (1,1)):
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
        canvas = self._canvas_append(canvas = canvas,
                                    add_point = (0,0),
                                    img = img)
        return canvas

    def _canvas_append(self,
                      canvas: np.ndarray,
                      add_point: np.ndarray,
                      img: np.ndarray,
                      mask: np.ndarray = None,
                      mask_label: int = None,
                      mode = "label"):
        """
        the actual working part, add a image to a canvas
        input:
          canvas: np.ndarray, 3-channel canvas
          add_point: tuple of int, the topleft point of the image to be added
          img: np.ndarray, 3-channel, the image to be added
          mask: np.ndarray(if it's there), 1-channel, the mask with the canvas
          mask_label: int(if mask if np.ndarray), the value of this label onto the mask
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

    def _try_insert(self,
                   img: np.ndarray,
                   canvas: np.ndarray,
                   mask: np.ndarray,
                   label: int,
                   patience: int,
                   mode = "label"):
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

    def _multinomial_distribution(self, prob_distribution):
        """
        Just a wrapper, I don't even know np and scipy's multinomial sampling is that weird...
        input:
          prob_distribution: any iterable, the multinomial distribution probability
        output:
          a sample from probability provided
        """
        return np.nonzero(np.random.multinomial(1,prob_distribution))[0][0]

    def crop_image(self, img):
        """
        crop all the 0 paddings aside
        input:
          img: np.ndarray, image to be cropped
        return:
          img: np.ndarray, image cropped
        """
        if len(img.shape) == 3:
            x = np.nonzero(np.any(img, axis = (0,2)))[0]
            y = np.nonzero(np.any(img, axis = (1,2)))[0]
        else:
            x = np.nonzero(np.any(img, axis = (0)))[0]
            y = np.nonzero(np.any(img, axis = (1)))[0]
        xs,xf = x[0],x[-1]
        ys,yf = y[0],y[-1]
        img = img[ys:yf,xs:xf]
        return img

    def generate_background(self,
                            canvas_size,
                            scanning_size = (50,50),
                            sample_threshold = 0.2,
                            offset_const = 10,
                            gaussian_variance = 0.01,
                            filter_size = 3):
        # the sampling is based on a grid-scanning scheme, we made some randomness on the grid scanning
        scan_sample = np.ceil(np.true_divide(self.example_img.shape[:2],scanning_size)).astype(int)
        non_zero_list = []
        # for the randomness, make some retry
        for offset in tqdm([(scanning_size[0]//offset_const*ratio,scanning_size[1]//offset_const*ratio) for ratio in range(offset_const)],
                           desc = "sampling",
                           leave = False):
          #search the grid
          for i in range(scan_sample[0]):
            for j in range(scan_sample[1]):
              # find the starting point, add some turbulence to the sample
              x = int(np.random.normal(loc = i, scale = gaussian_variance)*scanning_size[0]) + offset[0]
              y = int(np.random.normal(loc = j, scale = gaussian_variance)*scanning_size[1]) + offset[1]
              scanning_area = self.example_img[y : np.min((self.example_img.shape[1],y + scanning_size[1])),
                                               x : np.min((self.example_img.shape[0],x + scanning_size[0]))]
              # if the nonzero-area ratio is greater than the thereshold
              if np.sum(np.any(scanning_area, axis = 2))/(scanning_size[0]*scanning_size[1]) > sample_threshold:
                # sample it, crop possible additional black pad
                non_zero_list.append(self.crop_image(scanning_area))

        canvas = np.zeros((*canvas_size,3))
        # do something similar, add the image to grid
        scan_add = np.ceil(np.true_divide(canvas_size,scanning_size)).astype(int)
        # for this time we will pile the images for several layers, with some specific offset
        for offset in tqdm([(scanning_size[0]//offset_const*ratio,scanning_size[1]//offset_const*ratio) for ratio in range(offset_const)],
                           desc = "background generating",
                           leave = False):
          for i in range(scan_add[0]):
            for j in range(scan_add[1]):
              # when adding the image still provide some randomness
              add_point = (int(np.random.normal(loc = j, scale = gaussian_variance)*scanning_size[1])+offset[1],
                           int(np.random.normal(loc = i, scale = gaussian_variance)*scanning_size[0])+offset[0])
              try:
                canvas = self._canvas_append(canvas = canvas,
                                             #img = non_zero_list[np.random.randint(len(non_zero_list))],
                                             img = self.datagen.random_transform(non_zero_list[np.random.randint(len(non_zero_list))]),
                                             add_point = add_point)
              except:
                continue
        #convert all the black part to white
        canvas = np.where(canvas == 0, 255, canvas)
        # give it a filter to eliminate the clear edge
        filtered = Img.fromarray(canvas.astype("uint8")).filter(ImageFilter.MedianFilter(size = filter_size))
        filtered = np.array(filtered.filter(ImageFilter.MedianFilter(size = filter_size)), dtype = "uint8")
        return filtered

    def cut_circle_edges(self, mask, center = (1, 1), radius = 3, circle_mask_label = 100):
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

    def build_cluster(self, sub_canvas_size = (500,500)):
        """
        create a glomerus-proximal tubules cluster, with its mask
        """
        _circle_mask_label = 10
        _center = np.random.normal(loc = np.divide(sub_canvas_size,2),
                                  scale = np.divide(sub_canvas_size,50)).astype("int")
        _sub_canvas = np.zeros((*sub_canvas_size,3), dtype = "uint8")
        _mask = np.zeros(sub_canvas_size)

        _mask = self.cut_circle_edges(mask = _mask,
                                      center = _center,
                                      radius = sub_canvas_size[0]/2,
                                      circle_mask_label = _circle_mask_label)

        _glom_list = self.image_list[self.label_dict["glomerulus"]-1]
        _tubules_list = self.image_list[self.label_dict["proximal_tubule"]-1]

        _glom_chosen = _glom_list[np.random.randint(0,len(_glom_list))]
        _canvas, _mask = self._canvas_append(canvas= _sub_canvas,
                                             add_point= np.subtract(_center,
                                                                    np.divide(_glom_chosen.shape[:2],2)).astype("int"),
                                             img = _glom_chosen,
                                             mask = _mask,
                                             mask_label = self.label_dict["glomerulus"])
        for i in range(50):
            self._try_insert(img = _tubules_list[np.random.randint(0,len(_tubules_list))],
                             canvas = _sub_canvas,
                             mask = _mask,
                             label = self.label_dict["proximal_tubule"],
                             patience = 20)
        _mask = np.where(_mask == _circle_mask_label, 0, _mask)
        return self.crop_image(_sub_canvas), self.crop_image(_mask)

    def generate(self,
                 item_num: int = 5,
                 ratio_dict: dict = {"cluster":0.2, "artery": 0.5, 'arteriole': 0.3},
                 background_color = True,
                 return_dict = True
                 ):
        """
        the method to generate a new 3-channel collage and a mask
        input:
          item_num: int, the total number of items in this image
          ratio_list: list[float], the ratio of each cass, the number must be same to the labels
          background_color: bool, if True, will add a background color based on self.example_image
          return_dict: bool, if true, will return the dictionary of the generator
        return:
          _canvas: np.ndarray, self._canvas_size shape, 3 channel, the canvas with images added
          _mask, np.ndarray, self._canvas_size shape, 1 channel, the mask of the canvas
          label_dict: dict, the dictionary of label name and label value
        """
        # give the correct order, normalize the ratio
        _ratio_list = np.fromiter((ratio_dict[i] for i in ["cluster", "artery", 'arteriole']), dtype = float)
        _ratio_list = _ratio_list / np.sum(_ratio_list)

        _temp_canvas_size = self._canvas_size + 2*self.max_component_size
        # -------------------------------------Background----------------------------------------
        # generate a larger main canvas and a larger main mask, use them as the background of the final output
        if background_color == True:
          _canvas = self.generate_background(canvas_size = _temp_canvas_size)
        else:
          _canvas = np.full(shape = (_temp_canvas_size[0], _temp_canvas_size[1],3),
                            fill_value = 255,
                            dtype="uint8")
        _mask = np.zeros((_temp_canvas_size[0], _temp_canvas_size[1]))
        # -------------------------------------Component------------------------------------------
        # -------------------------------------Cluster--------------------------------------------
        _n_cluster = np.ceil(_ratio_list[0]*item_num)
        for _num_count in tqdm(np.arange(_n_cluster),
                               desc = "Appending Clusters...",
                               leave = False):
            _cluster_canvas, _cluster_mask = self.build_cluster(sub_canvas_size = self.cluster_size)
            _canvas, _mask = self._try_insert(img = _cluster_canvas,
                                              canvas = _canvas,
                                              mask = _mask,
                                              label = _cluster_mask,
                                              patience = self.patience,
                                              mode = "pattern")
        # -------------------------------artery, and arteriole------------------------------------
        # the rest ratio will be used for random selection
        _ratio_list = _ratio_list[1:]
        _ratio_list = _ratio_list / np.sum(_ratio_list)
        _item_list = ["artery", 'arteriole']
        #for the rest iteration
        for _num_count in tqdm(np.arange(item_num - _n_cluster),
                               desc = "Appending artery and arteriole...",
                               leave = False):
            # generate a random class from the distribution
            _class_add = self._multinomial_distribution(_ratio_list)
            _label_value = self.label_dict[_item_list[_class_add]]
            # if it's artery or arteriole, find the image_list for this class
            _image_list = self.image_list[_label_value - 1]
            # if the class is empty, skip
            if len(_image_list) == 0:
              pass
            # if it's not empty
            else:
              # choose a random images
              _img = _image_list[np.random.randint(len(_image_list))]
              _img_transformed = self.datagen.random_transform(_img)
              # append it to the canvas
              _canvas, _mask = self._try_insert(img = _img_transformed,
                                                canvas = _canvas,
                                                mask = _mask,
                                                label = _label_value,
                                                patience = self.patience)
        # -------------------------------------distal tubules-------------------------------------
        _image_list = self.image_list[self.label_dict["distal_tubule"]-1]
        for _num_count in tqdm(np.arange(1000),
                               desc = "Appending distal_tubules...",
                               leave = False):
            _img = _image_list[np.random.randint(len(_image_list))]
            _img_transformed = _img#self.datagen.random_transform(_img)
            # append it to the canvas
            _canvas, _mask = self._try_insert(img = _img_transformed,
                                              canvas = _canvas,
                                              mask = _mask,
                                              label = self.label_dict["distal_tubule"],
                                              patience = self.patience)

        # ------------------------------------PostProcessing----------------------------------------
        _cutbound_x = (int(self.canvas_size[0]/2),int(self.canvas_size[0]/2)+self.canvas_size[0])
        _cutbound_y = (int(self.canvas_size[1]/2),int(self.canvas_size[1]/2)+self.canvas_size[1])
        _cut_canvas = _canvas[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]]
        # add a gaussian noise
        _cut_canvas = _cut_canvas + (np.random.randn(*(_cut_canvas.shape))*self.gaussian_noise_constant)

        #make sure the output value is legal
        _cut_canvas = np.clip(_cut_canvas, a_min = 0, a_max = 255).astype("uint8")
        _cut_mask = _mask[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]].astype("uint8")
        if return_dict:
            return _cut_canvas, _cut_mask, self.label_dict
        else:
            return _cut_canvas, _cut_mask

    def generate_hdf5(self,
                      hdf5_dataset_fname: str = "save.h5",
                      length: int = 20,
                      item_num: int = 200,
                      vignettes_ratio_list: List[float] = [1.0,0.0],
                      background_color_ratio: float = 1.0):
        """
        a wrapper saving generated patches to hdf5 file, in current setting
        it will be saved into 4 different hdf5_file

        hdf5_dataset_fname: the example h5 filename, can be with or without .h5 extension
        length: int, the number of image generated
        item_num: int, most of time don't really need to work with
        vignettes_ratio_list: list of float, the ratio of each cass,
                                the number must be same to the labels.
                                but don't have to sum to 1
        background_color_ratio: float, the probability of images generated are with backgound color

        generating:
           hdf5 file, image with binary mask for each class specified in the file name.
        """
        assert length > 0
        assert 0 <= background_color_ratio <= 1
        if hdf5_dataset_fname[-3:] == '.h5':
            hdf5_dataset_fname = hdf5_dataset_fname[:-3]

        # generate enough collage and mask
        _collage_list = []
        _mask_list = []
        for _ctr in tqdm(range(length), desc = "generating...", leave = False):
            _collage, _mask= self.generate(ratio_list = vignettes_ratio_list,
                                           background_color = \
                                           np.random.binomial(size=1,
                                                              n=1,
                                                              p=background_color_ratio)[0],
                                           return_dict = False
                                           )
            _collage_list.append(_collage)

            _mask_list.append(_mask)

        _collage_list = np.stack(_collage_list, axis = 0)
        _mask_list = np.stack(_mask_list, axis = 0)

        _img_dtype = tables.UInt8Atom()
        _filename_dtype = tables.StringAtom(itemsize=255)
        _img_shape = (*self.canvas_size, 3)
        _mask_shape = self.canvas_size # mask is just a 2D matrix

        for label in tqdm(self.label_list, desc = "saving...", leave = False):
            # if use later version maybe save by _mask_list is more space-efficient
            _sub_mask_list = np.where(_mask_list == self.label_dict[label], 1, 0)
            with tables.open_file(f"{hdf5_dataset_fname}_{label}.h5", mode='w') as _hdf5_file:

                # use blosc compression
                filters = tables.Filters(complevel=1, complib='blosc')

                # filenames, images, masks are saved as three separate
                # earray in the hdf5 file tree

                _src_img_fnames = _hdf5_file.create_earray(
                    _hdf5_file.root,
                    name="src_image_fname",
                    atom=_filename_dtype,
                    shape=(0, ))

                _img_array = _hdf5_file.create_earray(
                    _hdf5_file.root,
                    name="img",
                    atom=_img_dtype,
                    shape=(0, *_img_shape),
                    chunkshape=(1, *_img_shape),
                    filters=filters)

                _mask_array = _hdf5_file.create_earray(
                    _hdf5_file.root,
                    name="mask",
                    atom=_img_dtype,
                    shape=(0, *_mask_shape),
                    chunkshape=(1, *_mask_shape),
                    filters=filters)

                # append newly created patches for every pair image and mask
                # and flush them incrementally to the hdf5 file
                _img_array.append(_collage_list)
                _mask_array.append(_sub_mask_list)
                _src_img_fnames.append([f'collage_generator_{time.strftime("%Y-%m-%d")}_{i}' \
                                        for i in range(_collage_list.shape[0])])
