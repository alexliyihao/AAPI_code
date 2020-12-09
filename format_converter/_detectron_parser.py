from PIL import Image
import json
import os
import glob
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import math
from detectron2.structures import BoxMode
from tqdm.notebook import tqdm
import torch

class _detectron_parser():

    def save_sliding_window(self,
                            collage,
                            mask,
                            color_dict,
                            image_name,
                            saving_path = "output",
                            window_size = np.array([1024,1024]),
                            offset = np.array([256,256])):
        scan = np.floor_divide(np.array(mask.shape[:2]) - window_size, offset)
        os.makedirs(saving_path, exist_ok=True)
        for i in range(scan[0]):
            for j in range(scan[1]):
                starting_point = (i*offset[0], j*offset[1])

                window_collage = collage[starting_point[1]:starting_point[1]+window_size[1], starting_point[0]:starting_point[0]+window_size[0]]
                window_mask = mask[starting_point[1]:starting_point[1]+window_size[1], starting_point[0]:starting_point[0]+window_size[0]]

                Image.fromarray(window_collage).save(os.path.join(saving_path, f"{image_name}-{i}-{j}-collage.png"))
                Image.fromarray(window_mask).save(os.path.join(saving_path, f"{image_name}-{i}-{j}-mask.png"))
        with open(os.path.join(saving_path, f"{image_name}_color_dict.json"), "w") as output:
            json.dump(color_dict, fp=output)

    def parse_detectron(self, path):
        """
        the final operator parse the images in a path
        """
        # These ids will be automatically increased as we go
        image_id = 0
        mask_path = glob.glob(os.path.join(path, "*mask.png"))
        formal_output = []
        # Create the annotations for each image
        for i in tqdm(range(len(mask_path)), desc = "parsing", leave = True):
            with open(os.path.join(path, os.path.basename(mask_path[i]).split("-")[0]+"_color_dict.json")) as input:
                color_dict = json.load(input)
            mask_image = Image.open(mask_path[i])
            annotations = []
            sub_masks = self._create_sub_masks(mask_image)
            for color, sub_mask in sub_masks.items():
                try:
                    category_id = color_dict[color]
                    annotation = self._create_sub_mask_annotation(sub_mask, image_id, category_id)
                    annotations.append(annotation)
                except:
                    pass
            formal_individual_format = {
                'annotations': annotations,
                'file_name': mask_path[i].replace("mask", "collage"),
                'height': mask_image.size[1],
                'image_id': image_id,
                'width': mask_image.size[0]}
            image_id += 1
            formal_output.append(formal_individual_format)
        return formal_output

    def parse_detectron_ROI(self, path):
        """
        the final operator parse the images in a path
        """
        # These ids will be automatically increased as we go
        image_id = 0
        mask_path = glob.glob(os.path.join(path, "*mask*.png"))
        formal_output = []
        # Create the annotations for each image
        for i in tqdm(range(len(mask_path)), desc = "parsing", leave = True):
            with open(mask_path[i].replace("mask", "color_dict").replace("png","json")) as input:
                color_dict = json.load(input)
            mask_image = Image.open(mask_path[i])
            annotations = []
            sub_masks = self._create_sub_masks(mask_image)
            for color, sub_mask in sub_masks.items():
                try:
                    category_id = color_dict[color]
                    annotation = self._create_sub_mask_annotation(sub_mask, image_id, category_id)
                    annotations.append(annotation)
                except:
                    pass
            formal_individual_format = {
                'annotations': annotations,
                'file_name': mask_path[i].replace("mask", "collage"),
                'height': mask_image.size[1],
                'image_id': image_id,
                'width': mask_image.size[0]}
            image_id += 1
            formal_output.append(formal_individual_format)
        return formal_output

    def _create_sub_mask_annotation(self, sub_mask, image_id, category_id):
        # Find contours (boundary lines) around each sub-mask
        # Note: there could be multiple contours if the object
        # is partially occluded. (E.g. an elephant behind a tree)
        contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            if len(segmentation) >= 6 and len(segmentation) % 2 == 0:
                segmentations.append(segmentation)

        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        bbox = [max(0,math.floor(x)), max(0,math.floor(y)), math.ceil(max_x), math.ceil(max_y)]
        annotation = {
            'bbox': bbox,
            'bbox_mode': BoxMode.XYXY_ABS,
            'segmentation': segmentations,
            'category_id': category_id,
        }

        return annotation

    def _create_sub_masks(self, mask_image):
        width, height = mask_image.size

        # Initialize a dictionary of sub-masks indexed by RGB colors
        sub_masks = {}
        for x in range(width):
            for y in range(height):
                # Get the RGB values of the pixel
                pixel = mask_image.getpixel((x,y))[:3]

                # If the pixel is not black...
                if pixel != (0, 0, 0):
                    # Check to see if we've created a sub-mask...
                    pixel_str = str(pixel)
                    sub_mask = sub_masks.get(pixel_str)
                    if sub_mask is None:
                       # Create a sub-mask (one bit per pixel) and add to the dictionary
                        # Note: we add 1 pixel of padding in each direction
                        # because the contours module doesn't handle cases
                        # where pixels bleed to the edge of the image
                        sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                    # Set the pixel value to 1 (default is 0), accounting for padding
                    sub_masks[pixel_str].putpixel((x+1, y+1), 1)

        return sub_masks

    def save_coco_raw(self, collage, mask, color_dict, root_path, name):
        """

        save {mask}, {collage} and {color_dict} in {path} with {name}
        args:
            collage: H*W*3 np.ndarray object, the actual collage
            mask: H*W*3 np.ndarray object, the color mask
            color_dict: {"(R,G,B)": label} dict
            path: the path under where the info to be saved
            name: the name template of the files
        """
        os.makedirs(root_path, exist_ok= True)
        Image.fromarray(mask).save(os.path.join(root_path, f"mask_{name}.png"))
        Image.fromarray(collage).save(os.path.join(root_path, f"collage_{name}.png"))
        with open(os.path.join(root_path, f"color_dict_{name}.json"),"w") as output:
            json.dump(color_dict, fp=output)

    def read_coco_raw(self, root_path, name):
        """
        load {mask}, {collage} and {color_dict} with {name} from {path}
        args:
            path: the path under where the info to be saved
            name: the name template of the files
        returns:
            collage: H*W*3 np.ndarray object, the actual collage
            mask: H*W*3 np.ndarray object, the color mask
            color_dict: {"(R,G,B)": label} dict
        """
        collage = np.array(Image.open(os.path.join(root_path, f"collage_{name}.png")))
        mask = np.array(Image.open(os.path.join(root_path, f"mask_{name}.png")))
        with open(os.path.join(root_path, f"color_dict_{name}.json"), "r") as readin:
            color_dict = json.load(readin)
        return collage, mask, color_dict

    def _extract_mask(self, pred_instance, label):
        """
        helper function to parse_np_masks,
        extract mask for a specific label from Detectron pred instance
        for GPU memory consideration use a loop here.
        Args:
            pred_instance: detectron2.structures.instances.Instances object,
                           can be obtained from predictor(image)["instances"]
            label: the label you want
        return:
            mask: 2d np.ndarray with dtype boolean, the boolean mask of {label}
        """
        pixel_mask = np.zeros(pred_instance.image_size, dtype=np.bool)
        for instance_mask in pred_instance[pred_instance.pred_classes == label].pred_masks:
            instance_mask = instance_mask.cpu().numpy()
            pixel_mask = np.logical_or(pixel_mask, instance_mask)
        return pixel_mask

    def parse_np_masks(self, pred_instance):
        """
        parse a pred_instance into a pixel mask with 1:glom, 2: artery, 3: tubules
        the importance is considered as glom > vessels > tubules
        Args:
            pred_instance: detectron2.structures.instances.Instances object,
                           can be obtained from predictor(image)["instances"]
        Return:
            mask: 2d np.ndarray with dtype uint8(image), the boolean mask of all labels with
                  1:glom, 2: artery and arteriole, 3: tubules
        """
        mask = np.zeros(pred_instance.image_size, dtype=np.uint8)

        arteriole_mask = self._extract_mask(pred_instance = pred_instance, label = 1)
        artery_mask = self._extract_mask(pred_instance = pred_instance, label = 2)
        distal_tubules_mask = self._extract_mask(pred_instance = pred_instance, label = 3)
        glom_mask = self._extract_mask(pred_instance = pred_instance, label = 4)
        proximal_tubules_mask = self._extract_mask(pred_instance = pred_instance, label = 5)

        mask = np.where(glom_mask, 1, mask)

        # when mask is not specified as glomerulus but can be specified as vessels:
        mask = np.where(
          np.logical_and(
            np.logical_or(arteriole_mask,artery_mask),
            np.logical_not(mask)
            ),
            2, mask)

        # otherwise, specified as tubules
        mask = np.where(
          np.logical_and(
            np.logical_or(distal_tubules_mask, proximal_tubules_mask),
            np.logical_not(mask)
            ),
            3, mask)

        return mask

    def save_formal_output(self, formal_output, path):
        '''
        save the list produced by parse_detectron_ROI as a json file
        Args: 
            formal_output: the list produced by parse_detectron_ROI

            path: the path of the saved json file
        '''
        formal_output_dict = {'list':formal_output}
        with open(path, 'w') as f:
            #BoxMode.XYXY_ABS will be dumped to 0
            json.dump(formal_output_dict, f)

    def load_formal_output(self, path, bbox_mode=BoxMode.XYXY_ABS):
        '''
        Load the list produced by parse_detectron_ROI from a json file
        Args: 
            path: the path of the saved json file

            bbox_mode: an object from BoxMode class
        Returns:
            the list of formal_output
        '''
        with open(path, 'r') as f:
            data = json.load(f)
        data = data['list']
        for i, _img_data in enumerate(data):
            _annotations = _img_data['annotations']
            for j in range(len(_annotations)):
                data[i]['annotations'][j]['bbox_mode'] = bbox_mode
        return data
