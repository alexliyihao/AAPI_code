import PIL.Image as Img
import numpy as np
from tqdm.notebook import tqdm
from PIL import ImageFilter
import tables
import time
import gc

"""
all the generative fuction of the collage_generator
"""

class _generative():
    def _generate_background(self,
                             canvas_size,
                             scanning_size = (50,50),
                             sample_threshold = 0.2,
                             offset_const = 10,
                             gaussian_variance = 0.01,
                             filter_size = 3):
        """
        generating the backgound based on the self._example_img
        args:
            canvas_size: pair of int, the canvas_size
            scanning_size: scanning_size pair of int, the size for moving scanning window
            sample_threshold: the ratio of nonzero area to the whole scanning window
            offset_const: the offset, the scanning window will move by step length of scanning_size/offset_const
            gaussian_variance: the randomness gives in the scanning window
            filter_size: the filter kernel size used
        return:
            filtered: the background for the canvas based on self._example_img
        """
        # the sampling is based on a grid-scanning scheme, we made some randomness on the grid scanning
        scan_sample = np.ceil(np.true_divide(self._example_img .shape[:2],scanning_size)).astype(int)
        non_zero_list = []
        _transform = self._generate_augmentation(mode = "other")

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
                    scanning_area = self._example_img [y : np.min((self._example_img .shape[1],y + scanning_size[1])),
                                                     x : np.min((self._example_img .shape[0],x + scanning_size[0]))]
                    # if the nonzero-area ratio is greater than the thereshold
                    if np.sum(np.any(scanning_area, axis = 2))/(scanning_size[0]*scanning_size[1]) > sample_threshold:
                        # sample it, crop possible additional black pad
                        non_zero_list.append(self._crop_image(scanning_area))
        gc.collect()
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
                                             img = self._augment(
                                                    image = non_zero_list[np.random.randint(0, len(non_zero_list))],
                                                    transform = _transform),
                                             add_point = add_point,
                                             mode = "background")
              except:
                continue
        #convert all the black part to white
        canvas = np.where(canvas == 0, 255, canvas)
        # give it a filter to eliminate the clear edge
        filtered = Img.fromarray(canvas.astype("uint8")).filter(ImageFilter.MedianFilter(size = filter_size))
        filtered = np.array(filtered.filter(ImageFilter.MedianFilter(size = filter_size)), dtype = "uint8")
        gc.collect()
        return filtered

    def _build_cluster(self, sub_canvas_size = (500,500), format = "pixel"):
        """
        create a glomerus-proximal tubules cluster, with its mask
        args:
            sub_canvas_size: tuple of int, length = 2, the size of sub_canvas used for a cluster
            format: str, "pixel" or "COCO", the mask format
        return:
            canvas: H*W*C np.ndarray, the actual image of this cluster
            mask: H*W*C np.ndarray, the mask of this cluster
        """
        # a augmentation instance for this
        _transform = self._generate_augmentation(mode = "other")
        # for this label never collide with existed label
        _circle_mask_label = len(self._label_list)+1

        # select a center coord of the canvas, add some randomness
        _center = np.random.normal(loc = np.divide(sub_canvas_size,2),
                                  scale = np.divide(sub_canvas_size,50)).astype("int")
        # canvas and mask
        _sub_canvas = np.zeros((*sub_canvas_size,3), dtype = "uint8")
        _mask = np.zeros(sub_canvas_size)

        # cut a round mask for a round cluster shape
        _mask = self._cut_circle_edges(mask = _mask,
                                       center = _center,
                                       radius = sub_canvas_size[0]/2,
                                       circle_mask_label = _circle_mask_label)

        # have the glom added to the center
        _glom_chosen = self._augment(
                            image = self._random_select(label = "glomerulus"),
                            transform = _transform)

        _sub_canvas, _mask = self._canvas_append(canvas= _sub_canvas,
                                                add_point= np.subtract(_center,
                                                                        np.divide(_glom_chosen.shape[:2],2)).astype("int"),
                                                img = _glom_chosen,
                                                mask = _mask,
                                                mask_label = self._label_dict["glomerulus"],
                                                format = format)

        # have the proximal tubule around
        for i in tqdm(range(self._num_proximal_per_cluster),
                      desc = "Generating Clusters...",
                      leave = False):
            _sub_canvas, _mask = self._try_insert(img = self._augment(
                                                      image = self._random_select(label = "proximal_tubule"),
                                                      transform = _transform),
                                                  canvas = _sub_canvas,
                                                  mask = _mask,
                                                  label = self._label_dict["proximal_tubule"],
                                                  patience = self._patience,
                                                  format = format)
        # remove the round mask
        _mask = np.where(_mask == _circle_mask_label, 0, _mask)
        # remove the potential zero pad
        return self._crop_image(_sub_canvas), self._crop_image(_mask)

    def generate(self,
                 item_num: int = 5,
                 ratio_dict: dict = {"cluster":0.2, "artery": 0.5, 'arteriole': 0.3},
                 background_color = True,
                 return_dict = True,
                 format = "pixel"
                 ):
        """
        the method to generate a new 3-channel collage and a {format} format mask
        args:
          item_num: int, the total number of glomerulus + artery + arterioles in this image
          ratio_dict: dict[string: float], the ratio of each cass,
                      the key must be same to the labels but not necessarily sum to one
          background_color: bool, if True, will add a background color based on self.example_image
          return_dict: bool, if true, will return the dictionary of the generator
          format: str, "pixel" or "COCO", how the mask will be updates by new vignettes
                  in "pixel", each individual mask will be saved on the same dimension
                  if "COCO", an additional mask which can be a intermediate format
                  translating to COCO will be returned
        return:
          _canvas: np.ndarray, self._canvas_size shape, 3 channel, the canvas with images added
          mask: np.ndarray of 1 channel, the mask with img's label added.
                if format is "COCO", another 3 channel np.ndarray is returned as the instance mask
                used as the intermediate when parsed to Detectron 2 format
          if format == "COCO":
              color_dict: dict{string:int}, the dictionary of each color to the categories
          if return_dict == True
              label_dict: dict{string:int}, the dictionary of label name and label value

        """
        assert item_num >= 0
        assert format in ["COCO", "pixel"]

        # give the correct order, normalize the ratio
        _ratio_list = np.fromiter((ratio_dict[i] for i in ["cluster", "artery", 'arteriole']), dtype = float)
        _ratio_list = _ratio_list / np.sum(_ratio_list)

        _temp_canvas_size = self._canvas_size + 2*self._max_component_size
        # -------------------------------------Background----------------------------------------
        # generate a larger main canvas and a larger main mask, use them as the background of the final output
        _canvas = np.zeros(shape = (_temp_canvas_size[0], _temp_canvas_size[1],3), dtype="uint8")
        _mask = np.zeros((_temp_canvas_size[0], _temp_canvas_size[1]))
        # -------------------------------------Component------------------------------------------
        # -------------------------------------Cluster--------------------------------------------
        if format == "COCO":
            # the color used in COCO mask
            self.existed_color = []
            self.color_dict = {}
        _n_cluster = np.ceil(_ratio_list[0]*item_num)
        for _num_count in tqdm(np.arange(_n_cluster),
                               desc = "Appending Clusters...",
                               leave = False):

            _cluster_canvas, _cluster_mask = self._build_cluster(
                                                  sub_canvas_size = self._cluster_size,
                                                  format = format
                                                  )
            if _num_count == 0:
                _canvas, _mask = self._init_insert(
                                    img = _cluster_canvas,
                                    canvas = _canvas,
                                    mask = _mask,
                                    label = _cluster_mask,
                                    mode = "pattern",
                                    format = format
                                    )
            else:
                _canvas, _mask = self._secondary_insert(
                                    img = _cluster_canvas,
                                    canvas = _canvas,
                                    mask = _mask,
                                    label = _cluster_mask,
                                    patience = self._patience,
                                    mode = "pattern",
                                    format = format
                                    )
        # -------------------------------artery, and arteriole------------------------------------
        # the rest ratio will be used for random selection
        _transform = self._generate_augmentation(mode = "other")
        _ratio_list = _ratio_list[1:]
        _ratio_list = _ratio_list / np.sum(_ratio_list)
        _vessel_name_list = ["artery", 'arteriole']
        #for the rest iteration
        for _num_count in tqdm(np.arange(item_num - _n_cluster),
                               desc = "Appending artery and arteriole...",
                               leave = False):
            # generate a random class from the distribution， note this label is a string
            _label = _vessel_name_list[self._multinomial_distribution(_ratio_list)]

            # append it to the canvas
            _canvas, _mask = self._secondary_insert(
                             img = self._augment(
                                    image = self._random_select(label = _label),
                                    transform = _transform),
                             canvas = _canvas,
                             mask = _mask,
                             label = self._label_dict[_label],
                             patience = self._patience,
                             format = format
                             )
        # -------------------------------------distal tubules-------------------------------------
        _transform = self._generate_augmentation(mode = "distal_tubules")

        for _num_count in tqdm(np.arange(self._num_distal_per_image),
                               desc = "Appending distal_tubules...",
                               leave = False):

            # append it to the canvas
            _canvas, _mask = self._try_insert(img = self._augment(
                                                image = self._random_select(label = "distal_tubule"),
                                                transform = _transform),
                                              canvas = _canvas,
                                              mask = _mask,
                                              label = self._label_dict["distal_tubule"],
                                              patience = self._patience,
                                              format = format)

        # ------------------------------------PostProcessing----------------------------------------
        # cut the additional part of the canvas
        _cutbound_x = (self._max_component_size[0],self._max_component_size[0]+self._canvas_size[0])
        _cutbound_y = (self._max_component_size[1],self._max_component_size[1]+self._canvas_size[1])
        _canvas = _canvas[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]]
        _mask = _mask[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]].astype("uint8")
        gc.collect()
        if background_color == True:
          _background = self._generate_background(canvas_size = self._canvas_size)
          _canvas = np.where(_canvas != 0, _canvas, _background)
        else:
          _canvas = np.where(_canvas != 0, _canvas, 255)
        gc.collect()
        # add a gaussian noise
        _canvas = _canvas + (np.random.randn(*(_canvas.shape))*self._gaussian_noise_constant)

        #make sure the output value is legal
        _canvas = np.clip(_canvas, a_min = 0, a_max = 255).astype("uint8")

        if format == "COCO":
            self._visualize_result(
                    collage = _canvas,
                    mask = _mask[:,:,0],
                    dictionary = self._label_dict)
            # refresh the used color list
            self.existed_color = []
            if return_dict:
                return _canvas, _mask[:,:,0], _mask[:,:,1:4], self.color_dict, self._label_dict,
            else:
                return _canvas, _mask[:,:,0], _mask[:,:,1:4], self.color_dict,
        else:
            self._visualize_result(
                    collage = _canvas,
                    mask = _mask,
                    dictionary = self._label_dict)
            if return_dict:
                return _canvas, _mask, self._label_dict
            else:
                return _canvas, _mask
