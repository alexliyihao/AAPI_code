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
        generating the backgound based on the self.example_img
        args:
            canvas_size: the canvas_size
            scanning_size: the size for moving window scanning
            sample_threshold: the ratio of nonzero area to the whole scanning window
            offset_const: the offset, the scanning window will move by step length of scanning_size/offset_const
            gaussian_variance: the randomness gives in the scanning window
            filter_size: the filter kernel size used
        return:
            filtered: the background for the canvas based on self.example_img
        """
        # the sampling is based on a grid-scanning scheme, we made some randomness on the grid scanning
        scan_sample = np.ceil(np.true_divide(self.example_img.shape[:2],scanning_size)).astype(int)
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
                    scanning_area = self.example_img[y : np.min((self.example_img.shape[1],y + scanning_size[1])),
                                                     x : np.min((self.example_img.shape[0],x + scanning_size[0]))]
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
                                             img = self._augment(image = self._random_select(non_zero_list),
                                                                _transform = _transform),
                                             add_point = add_point)
              except:
                continue
        #convert all the black part to white
        canvas = np.where(canvas == 0, 255, canvas)
        # give it a filter to eliminate the clear edge
        filtered = Img.fromarray(canvas.astype("uint8")).filter(ImageFilter.MedianFilter(size = filter_size))
        filtered = np.array(filtered.filter(ImageFilter.MedianFilter(size = filter_size)), dtype = "uint8")
        gc.collect()
        return filtered

    def _build_cluster(self, sub_canvas_size = (500,500), format = "pixel", existed_color = None):
        """
        create a glomerus-proximal tubules cluster, with its mask
        args:
            sub_canvas_size: tuple of int, length = 2, the size of sub_canvas used for a cluster
        """
        # a augmentation instance for this
        _transform = self._generate_augmentation(mode = "other")
        # for this label never collide with existed label
        _circle_mask_label = len(self.label_list)+1


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

        _glom_list = self.image_list[self.label_dict["glomerulus"]-1]
        _tubules_list = self.image_list[self.label_dict["proximal_tubule"]-1]
        # have the glom added to the center
        _glom_chosen = self._augment(image = self._random_select(_glom_list),
                                     _transform = _transform)
        _sub_canvas, _mask = self._canvas_append(canvas= _sub_canvas,
                                                add_point= np.subtract(_center,
                                                                        np.divide(_glom_chosen.shape[:2],2)).astype("int"),
                                                img = _glom_chosen,
                                                mask = _mask,
                                                mask_label = self.label_dict["glomerulus"],
                                                format = format,
                                                existed_color = existed_color)
        # have the proximal tubule around
        for i in tqdm(range(500),
                      desc = "Generating Clusters...",
                      leave = False):
            _sub_canvas, _mask = self._try_insert(img = self._augment(image = self._random_select(_tubules_list),
                                                      _transform = _transform),
                                                    canvas = _sub_canvas,
                                                    mask = _mask,
                                                    label = self.label_dict["proximal_tubule"],
                                                    patience = self.patience,
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
        the method to generate a new 3-channel collage and a mask
        args:
          item_num: int, the total number of items in this image
          ratio_dict: dict[string: float], the ratio of each cass,
                      the key must be same to the labels but not necessarily sum to one
          background_color: bool, if True, will add a background color based on self.example_image
          return_dict: bool, if true, will return the dictionary of the generator
          format: str, "pixel" or "COCO", how the mask will be updates by new vignettes
                  in "pixel", each individual mask will be saved on the same dimension
                  if "COCO", each individual mask will be saved on a additional channel
        return:
          _canvas: np.ndarray, self.canvas_size shape, 3 channel, the canvas with images added
          mask: if format is "pixel" np.ndarray of 1 channel, the mask with img's label added.
                if format is "COCO", np.ndarray of multi-channels, the mask with img's label added.
          label_dict: dict, the dictionary of label name and label value
        """

        # give the correct order, normalize the ratio
        _ratio_list = np.fromiter((ratio_dict[i] for i in ["cluster", "artery", 'arteriole']), dtype = float)
        _ratio_list = _ratio_list / np.sum(_ratio_list)

        _temp_canvas_size = self.canvas_size + 2*self.max_component_size
        # -------------------------------------Background----------------------------------------
        # generate a larger main canvas and a larger main mask, use them as the background of the final output
        if background_color == True:
          _canvas = self._generate_background(canvas_size = _temp_canvas_size)
        else:
          _canvas = np.full(shape = (_temp_canvas_size[0], _temp_canvas_size[1],3),
                            fill_value = 255,
                            dtype="uint8")
        _mask = np.zeros((_temp_canvas_size[0], _temp_canvas_size[1]))
        # -------------------------------------Component------------------------------------------
        # -------------------------------------Cluster--------------------------------------------
        if format == "COCO":
            self.existed_color = []
            self.color_dict = {}
        _n_cluster = np.ceil(_ratio_list[0]*item_num)
        for _num_count in tqdm(np.arange(_n_cluster),
                               desc = "Appending Clusters...",
                               leave = False):
            _cluster_canvas, _cluster_mask, existed_color = self._build_cluster(
                                                                sub_canvas_size = self.cluster_size,
                                                                format = format)
            if _num_count == 0:
                _canvas, _mask = self._init_insert(img = _cluster_canvas,
                                                   canvas = _canvas,
                                                   mask = _mask,
                                                   label = _cluster_mask,
                                                   mode = "pattern",
                                                   format = format)
            else:
                _canvas, _mask = self._secondary_insert(img = _cluster_canvas,
                                                        canvas = _canvas,
                                                        mask = _mask,
                                                        label = _cluster_mask,
                                                        patience = self.patience,
                                                        mode = "pattern",
                                                        format = format)
        # -------------------------------artery, and arteriole------------------------------------
        # the rest ratio will be used for random selection
        _transform = self._generate_augmentation(mode = "other")
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
              # append it to the canvas
              _canvas, _mask = self._secondary_insert(
                                 img = self._augment(image = self._random_select(_image_list),
                                                    _transform = _transform),
                                 canvas = _canvas,
                                 mask = _mask,
                                 label = _label_value,
                                 patience = self.patience,
                                 format = format
                                 )
        # -------------------------------------distal tubules-------------------------------------
        _transform = self._generate_augmentation(mode = "distal_tubules")
        _image_list = self.image_list[self.label_dict["distal_tubule"]-1]
        for _num_count in tqdm(np.arange(3000),
                               desc = "Appending distal_tubules...",
                               leave = False):

            # append it to the canvas
            _canvas, _mask = self._try_insert(img = self._augment(image = self._random_select(_image_list),
                                                                 _transform = _transform),
                                              canvas = _canvas,
                                              mask = _mask,
                                              label = self.label_dict["distal_tubule"],
                                              patience = self.patience,
                                              format = format)

        # ------------------------------------PostProcessing----------------------------------------
        # cut the additional part of the canvas
        _cutbound_x = (self.max_component_size[0],self.max_component_size[0]+self.canvas_size[0])
        _cutbound_y = (self.max_component_size[1],self.max_component_size[1]+self.canvas_size[1])
        _canvas = _canvas[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]]
        _mask = _mask[_cutbound_x[0]:_cutbound_x[1],_cutbound_y[0]:_cutbound_y[1]].astype("uint8")
        gc.collect()
        # add a gaussian noise
        _canvas = _canvas + (np.random.randn(*(_canvas.shape))*self.gaussian_noise_constant)

        #make sure the output value is legal
        _canvas = np.clip(_canvas, a_min = 0, a_max = 255).astype("uint8")

        if format == "COCO":
            self.existed_color = []
            if return_dict:
                return _canvas, _mask[:,:,1:4], self.color_dict, self.label_dict
            else:
                return _canvas, _mask[:,:,1:4], self.color_dict
        else:
            if return_dict:
                return _canvas, _mask, self.label_dict
            else:
                return _canvas, _mask


    def generate_hdf5(self,
                      hdf5_dataset_fname: str = "save.h5",
                      image_num: int = 20,
                      item_num: int = 5,
                      vignettes_ratio_dict: dict = {"cluster":0.2, "artery": 0.5, 'arteriole': 0.3},
                      background_color_ratio: float = 1.0):
        """
        a wrapper saving generated patches to hdf5 file, in current setting
        the mask will be saved into individual binary mask

        hdf5_dataset_fname: the example h5 filename, can be with or without .h5 extension
        image_num: int, the number of image generated
        item_num: int, most of time don't really need to work with
        vignettes_ratio_dict: dict[string: float], the ratio of each cass,
                              key must be same to labels but not necessarily sum to one
        background_color_ratio: float, the probability of images generated are with backgound color

        generating:
           hdf5 file, image with binary mask for each class specified in the file name.
        """
        assert image_num > 0
        assert 0 <= background_color_ratio <= 1
        if hdf5_dataset_fname[-3:] == '.h5':
            hdf5_dataset_fname = hdf5_dataset_fname[:-3]

        # generate enough collage and mask
        _collage_list = []
        _mask_list = []
        for _ctr in tqdm(range(image_num), desc = "generating...", leave = False):
            _collage, _mask= self.generate(
                item_num = item_num,
                ratio_dict = vignettes_ratio_dict,
                background_color = np.random.binomial(size=1,n=1,p=background_color_ratio)[0],
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
