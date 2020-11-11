import datetime
import json
import os
import re
import fnmatch
import PIL.Image as Img
import numpy as np
from pytz import timezone
import json
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

class _coco_converter():

    def generate_info():
        """
        hardcoded
        """
        return {
            "description": "AAPI Dataset",
            "url": "https://github.com/Auto-annotation-of-Pathology-Images",
            "version": "0.1.0",
            "year": 2020,
            "contributor": "AAPI group",
            "date_created": datetime.datetime.now(timezone("EST")).isoformat(' ')
        }

    def generate_license():
        """
        hardcoded
        """
        return  [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

    def generate_categories():
        """
        hardcoded, maybe we can do some extension?
        """
        return [{'color': [0, 255, 0],
                'id': 1,
                'isthing': 1,
                'name': 'arteriole',
                'supercategory': 'vessels'},
               {'color': [0, 0, 255],
                'id': 2,
                'isthing': 1,
                'name': 'artery',
                'supercategory': 'vessels'},
               {'color': [255, 255, 0],
                'id': 3,
                'isthing': 1,
                'name': 'distal_tubule',
                'supercategory': 'tubules'},
               {'color': [255, 0, 128],
                'id': 4,
                 'isthing': 1,
                'name': 'glomerulus',
                'supercategory': 'glomerulus'},
               {'color': [255, 100, 0],
                'id': 5,
                'isthing': 1,
                'name': 'proximal_tubule',
                'supercategory': 'tubules'}]

    def parse_masks(mask):
        """
        parse a coco-intermediate format mask into COCO color mask,
        with the color dict for dict

        args:
            mask: np.ndarray,coco-intermediate format mask
        returns:
            image: PIL image object, a image using different color as Instance segmentation label
            color_dict: dict of {'(R,G,B): label'} as semantic segmentation label
        """
        colors = random_color(mask.shape[2]-1)
        image = np.zeros((*mask.shape[:2],3))
        color_dict = {}
        for i in np.arange(1,mask.shape[2]):
            sub_mask = mask[:,:,i]
            color = colors[i-1]
            label = int(np.max(sub_mask))
            image[sub_mask.astype(bool)] = color
            color_dict[str(tuple(color))] = label
        return Img.fromarray(image.astype("uint8")), color_dict

    def save_intermediate(root_path, image_dir, seg_dir, image, segementation, color_dict):
        """
        save the intermediate format for preparing COCO
        """
        time_generate = datetime.datetime.now(timezone("EST")).isoformat(' ')
        image.save(fp = os.path.join(image_dir, f"image_{time_generate}.png"))
        segmentation.save(fp = os.path.join(seg_dir, f"mask_{time_generate}.png"))
        with open(os.path.join(seg_dir, f"mask_dict_{time_generate}.json"), "w") as open_file:
            json.dump(color_dict, open_file)

    def generate_annotation(image_id, file_name, segment_info):
        return {
                "image_id": image_id,
                "file_name": file_name,
                "segments_info": segment_info
                }

    def create_sub_masks(mask_image):
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
                        sub_masks[pixel_str] = Img.new('1', (width+2, height+2))

                    # Set the pixel value to 1 (default is 0), accounting for padding
                    sub_masks[pixel_str].putpixel((x+1, y+1), 1)

        return sub_masks

    def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id):
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

        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y

        annotation = {
            'iscrowd': 0,
            'category_id': category_id,
            'id': annotation_id,
            'bbox': (int(x), int(y), int(width), int(height)),
            'area': int(multi_poly.area)
        }
        return annotation

    def generate_image_info(image_path, image_id):
        image = Img.open(image_path)
        return {
            "id": image_id,
            "width": image.size[0],
            "height": image.size[1],
            "file_name": image_path,
            "license": 1,
            "flickr_url": "ungiven",
            "coco_url": "ungiven",
            "date_captured": os.path.splitext(image_path)[0][6:],
            }

    def generate_segment(mask_path, dict_path, image_id, annotation_id)
        mask = Img.open(mask_path)
        sub_masks = create_sub_masks(mask)

        with open(dict_path, "r") as open_file:
            mask_dict = json.load(open_file)

        segment_info = []
        for color, sub_mask in sub_masks.items():
            category_id = mask_dict[color]
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id)
            segment_info.append(annotation)
            annotation_id += 1
        return segment_info, annotation_id

    def save_COCO_format(self, image_dir, seg_dir):
        """
        save a generated coco_prepared format image, with it's annotation, as COCO format
        the json file will be saved in the direct parent folder of seg_dir
        """
        # this id should be the number of images in the direction
        image_info = []
        annotation_info = []
        annotation_id = 1
        images = sorted(os.listdir(image_dir))
        seg_masks = sorted([i for i in os.listdir(seg_dir) if "dict" not in i])
        seg_dicts = sorted([i for i in os.listdir(seg_dir) if "dict" in i])

        for image_id, image_path, mask_path, dict_path in zip(np.arange(len(images))+1, images, seg_masks,seg_dicts):
            image_info.append(generate_image_info(image_path = image_path, image_id = image_id))

            segment_info, annotation_id = generate_segment(mask_path, dict_path, image_id, annotation_id)
            annotation_info.append(
                generate_annotation(image_id = image_id,
                                    file_name = mask_path,
                                    segment_info = segment_info)
                )
        coco_output = {"info": generate_info(),
                       "images": image_info,
                       "annotations": annotation_info,
                       "licenses": generate_license(),
                       "categories": generate_categories()}

        with open(f"{split_seg_dir}.json","w") as open_file:
            json.dump(coco_output, open_file)
