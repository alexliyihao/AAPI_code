from lxml import etree, objectify
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
import cv2
import numpy as np
from pandas import DataFrame
from PIL import ImageColor
from typing import Union


class ASAPAnnotation:
    """
    Custom data structure for storing polygon-like annotations with
    functionality to serialize into an xml block
    """

    def __init__(self, geometry, annotation_name, group_name, render_color="#F4FA58"):
        """

        Parameters
        ----------
        geometry : shapely.geometry.Polygon
            A shape described by shapely.geometry.Polygon object
        annotation_name : str or None
            name for this annotation
        group_name : str or None
            group name for this annotation;
            group block xml will be generated using this attribute
        render_color : str, HEX code for color, starts with #
            color used when rendered on ASAP frontend
        """
        self.geometry = geometry
        self.annotation_name = str(annotation_name)
        self.group_name = str(group_name)
        self.render_color = str(render_color)

    def to_xml(self):
        """
        Serialize this object to xml (etree.Element), using ASAP format
        Returns
        -------
        annotation_xml : etree.Element
            an lxml object describing the same object
        """
        coordinates = self.geometry.exterior.coords[:-1]
        coordinates_attrs = [{"Order": str(i), "X": str(x), "Y": str(y)}
                             for i, (x, y) in enumerate(coordinates)]

        annotation_xml = etree.Element("Annotation",
                                       Name=self.annotation_name,
                                       Type=self.geometry.geom_type,
                                       PartOfGroup=self.group_name,
                                       Color=self.render_color)

        coordinates_xml = etree.Element("Coordinates")

        for attrs in coordinates_attrs:
            inner_xml = etree.Element("Coordinate", **attrs)
            coordinates_xml.append(inner_xml)

        annotation_xml.append(coordinates_xml)

        return annotation_xml

    @staticmethod
    def generate_annotation_groups(annotations, parent_group_name=None):
        """
        Generate xml for annotation groups using a list of annotations.
        Group names are extracted from ASAPAnnotation.group_name attribute.

        Parameters
        ----------
        annotations : List[ASAPAnnotations]
            list of annotations
        parent_group_name : str or None
            Currently only support one layer of ancestor for every group

        Returns
        -------
        annotation_groups : List[etree.Element]
            list of xmls for annotation groups
        """
        unique_groups = set([a.group_name for a in annotations])
        annotation_groups = []

        for group_name in unique_groups:
            # use the color of first annotation in each group
            first_annotation = next(filter(lambda a: a.group_name == group_name, annotations))
            group_color = first_annotation.render_color

            group_xml = etree.Element("Group",
                                      Name=group_name,
                                      PartOfGroup=str(parent_group_name),
                                      Color=str(group_color))
            etree.SubElement(group_xml, "Attributes")
            # TODO: add attributes implementations; currently is a placeholder
            annotation_groups.append(group_xml)

        return annotation_groups

    def __repr__(self):
        return pretty_format_xml(self.to_xml())

    def __str__(self):
        return self.__repr__()


def pretty_format_xml(xml_node):
    return etree.tostring(xml_node, pretty_print=True).decode('UTF-8').strip()


def write_xml_to_file(filename, xml_node):
    tree = etree.ElementTree(xml_node)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding="utf-8")


def _hex_to_rgb(hex_code: str):
    return ImageColor.getcolor(hex_code, "RGB")


def load_annotations_from_asap_xml(xml_path):
    """
    Deserialize annotations into custom data structures from xml file
    Parameters
    ----------
    xml_path : Path or str
        path to ASAP format xml annotation file
    Returns
    -------
    asap_annotations : List[ASAPAnnotations]
        list of ASAPAnnotations after deserialization

    """
    with open(xml_path) as f:
        res = f.read().encode("utf-8")

    root = objectify.fromstring(res)

    # assume no recursive structure for annotation groups now (i.e., one layer only)
    # "true" group nodes are leaf nodes of root.AnnotationGroups
    xml_annotation_groups = list(root.AnnotationGroups.getchildren())
    group_names = set(map(lambda g: g.get("Name"), xml_annotation_groups))

    xml_annotations = list(root.Annotations.getchildren())
    asap_annotations = []
    for xml_annotation in xml_annotations:
        xy_coordinates = list(map(lambda c: (float(c.get("X")), float(c.get("Y"))),
                                  xml_annotation.Coordinates.getchildren()))
        geometry = Polygon(xy_coordinates)
        name = xml_annotation.get("Name")
        group_name = xml_annotation.get("PartOfGroup")
        color = xml_annotation.get("Color")

        assert xml_annotation.get("PartOfGroup") in group_names, \
            f"Found inconsistent group name {group_name} for annotation {name};\n" \
            f"Acceptable groups: {group_names}."

        asap_annotation = ASAPAnnotation(geometry=geometry,
                                         annotation_name=name,
                                         group_name=group_name,
                                         render_color=color)
        asap_annotations.append(asap_annotation)

    return asap_annotations


def load_annotations_from_halo_xml(xml_path):
    """
    Deserialize annotations into custom data structures from xml file (HALO format)
    Parameters
    ----------
    xml_path : Path or str
        path to ASAP format xml annotation file
    Returns
    -------
    asap_annotations : List[ASAPAnnotations]
        list of ASAPAnnotations after deserialization

    """
    with open(xml_path) as f:
        res = f.read().encode("utf-8")

    root = objectify.fromstring(res)

    annotation_groups = root.getchildren()

    asap_annotations = []
    for annotation_group in annotation_groups:
        group_name = annotation_group.get("Name")
        line_color = annotation_group.get("LineColor")
        group_color = "#{:06x}".format(int(line_color))

        regions = annotation_group.Regions.getchildren()
        for i, region in enumerate(regions):
            xy_coordinates = [(int(v.get("X")), int(v.get("Y")))
                              for v in region.Vertices.getchildren()]

            asap_annotation = ASAPAnnotation(geometry=Polygon(xy_coordinates),
                                             annotation_name=f"{group_name}_{i}",
                                             group_name=group_name,
                                             render_color=group_color)

            asap_annotations.append(asap_annotation)

    return asap_annotations


def create_asap_annotation_file(annotations, filename):
    """
    Dump list of annotations to file
    Parameters
    ----------
    annotations : List[ASAPAnnotation]
        list of annotations
    filename : PosixPath or str
        path to the target file

    """
    annotation_xmls = list(map(lambda x: x.to_xml(), annotations))
    group_xmls = ASAPAnnotation.generate_annotation_groups(annotations)

    root = etree.Element("ASAP_Annotations")
    annotation_root = etree.SubElement(root, "Annotations")
    annotation_group_root = etree.SubElement(root, "AnnotationGroups")

    list(map(lambda x: annotation_root.append(x), annotation_xmls))
    list(map(lambda x: annotation_group_root.append(x), group_xmls))

    write_xml_to_file(str(filename), root)


def annotation_to_mask(annotations,
                       label_info,
                       upper_left_coordinates,
                       mask_shape,
                       level,
                       level_factor=4):
    """
    Convert a polygon or multipolygon list back to an image mask ndarray;
    useful for training segmentation models

    Parameters
    ----------
    annotations : List[ASAPAnnotation]
        list of annotations with custom data structure
    label_info : DataFrame
        pandas DataFrame describing mappings from label id to label class name, color
        must contain the following three columns:
            1) label: int, id for this class
            2) label_name: str, class name for this class
            3) color: str, 6 digit hex code starts with "#",
                      describing the rendering color for the class
    upper_left_coordinates : tuple(float, float)
        coordinate of upper left point of the mask,
        defined in level-0 coordinate system of original slide
    mask_shape : tuple
        the exact shape of output mask, can be 2D or 3D;
        (not the original shape in slides, as the output level may not be level-0)
    level : int
        custom level to generate annotations, higher level indicates lower resolution
    level_factor : int
        downsampling factor between every two adjacent levels

    Returns
    -------
    img_mask : np.array
        generated mask using uint8 labels, with shape (mask_shape[0], mask_shape[1], 3)
    """

    if mask_shape[-1] != 3 and len(mask_shape) == 2:
        # add the third channel if needed
        mask_shape = [*mask_shape, 3]

    upper_left_x, upper_left_y = upper_left_coordinates
    ds_rate = level_factor ** level
    img_mask = np.zeros(mask_shape, np.uint8)

    for label_row in label_info.itertuples():
        # manipulate every label individually
        label_name = label_row.label_name

        polygons = [a.geometry for a in annotations if label_name == a.group_name]
        # see https://shapely.readthedocs.io/en/latest/manual.html#affine-transformations
        transform_matrix = [1/ds_rate, 0, 0, 1/ds_rate,
                            -upper_left_x/ds_rate, -upper_left_y/ds_rate]
        polygons = list(map(lambda p: affine_transform(p, transform_matrix), polygons))

        # function to round and convert to int
        round_coords = lambda x: np.array(x).round().astype(np.int32)
        exteriors = [round_coords(poly.exterior.coords) for poly in polygons]
        interiors = [round_coords(pi.coords) for poly in polygons
                     for pi in poly.interiors]
        cv2.fillPoly(img_mask, interiors, 0)
        cv2.fillPoly(img_mask, exteriors, [label_row.label]*3) # all three channels should be assigned

    return img_mask


def mask_to_annotation(mask,
                       label_info,
                       upper_left_coordinates,
                       level,
                       level_factor=4,
                       min_area=10):
    """
    Convert a mask (uint8 array-like) to list of annotations,
    in order to render on ASAP frontend
    Inspired from: https://michhar.github.io/masks_to_polygons_and_back/

    Parameters
    ----------
    mask : 3D np.array or PIL image
        uint8 numpy array mask, each unique label id stands for a unique class
    label_info : DataFrame
        pandas DataFrame describing mappings from label id to label class name, color
        must contain the following three columns:
            1) label: int, id for this class
            2) label_name: str, class name for this class
            3) color: str, 6 digit hex code starts with "#",
                      describing the rendering color for the class
    upper_left_coordinates : tuple(float, float)
        coordinate of upper left point of the mask,
        defined in level-0 coordinate system of original slide
    level : int
        custom level to generate annotations, higher level indicates lower resolution
    level_factor : int
        downsampling factor between every two adjacent levels
    min_area : int
        polygons with area smaller than min_area will be ignored for generating annotations

    Returns
    -------
    annotations: List[ASAPAnnotation]
        list of annotations with custom data structure
    """

    def mask_to_polygon(binary_mask):
        image, contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(binary_mask),
                                                      cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_TC89_L1)

        if not contours:
            return None

        # only use the shell to construct polygons,
        # since the frontend doesn't support annotations with holes
        all_polygons = []
        for idx, cnt in enumerate(contours):
            if cv2.contourArea(cnt) >= min_area:
                assert cnt.shape[1] == 1
                poly = Polygon(shell=cnt[:, 0, :])
                all_polygons.append(poly)
        return all_polygons

    annotations = []
    upper_left_x, upper_left_y = upper_left_coordinates
    ds_rate = level_factor ** level

    mask_2d = np.array(mask)[..., 0]
    for label_row in label_info.itertuples():

        if label_row.label == 0:
            # skip the background
            continue

        binary_mask = np.array(mask_2d == label_row.label, dtype=np.uint8)
        polygons = mask_to_polygon(binary_mask)
        if polygons is not None:
            # apply reverse transformation to original coordinates
            transform_matrix = [ds_rate, 0, 0, ds_rate, upper_left_x, upper_left_y]
            polygons = map(lambda p: affine_transform(p, transform_matrix), polygons)

            label_name = label_row.label_name
            annotations += [ASAPAnnotation(geometry=p,
                                           annotation_name=f"{label_name}_{i}",
                                           group_name=label_name,
                                           render_color=label_row.color)
                            for i, p in enumerate(polygons)]
    return annotations


def generate_colorful_mask(mask, label_info):
    """
    Visualize mask in a colorful manner;
    rendering color is derived from label_info dataframe

    Parameters
    ----------
    mask : np.array
        mask containing uint8 type labels
    label_info : DataFrame
        pandas DataFrame, containing mappings of label id to color

    Returns
    -------
    colorful_mask : np.array
        each label is visualized using a unique color

    """
    colorful_mask = np.copy(mask)
    mask_2d = mask[..., 0] if len(mask.shape) == 3 else mask

    for row in label_info.itertuples():
        label, color = row.label, row.color
        colorful_mask[mask_2d == label] = _hex_to_rgb(color)

    return colorful_mask
