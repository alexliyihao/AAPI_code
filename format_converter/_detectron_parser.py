from PIL import Image
import json
import os
import glob
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import math
from detectron2.structures import BoxMode

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

    def parse_coco(self, path):
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
