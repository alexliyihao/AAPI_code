from lxml import etree, objectify
from shapely.geometry import Polygon


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
    def generate_annotation_groups(annotations, parent_group_name=None, group_color=None):
        """
        Generate xml for annotation groups using a list of annotations.
        Group names are extracted from ASAPAnnotation.group_name attribute.

        Parameters
        ----------
        annotations : List[ASAPAnnotations]
            list of annotations
        parent_group_name : str or None
            Currently only support one layer of ancestor for every group
        group_color : str, HEX code for color, starts with #
            color used when rendered on ASAP frontend
        Returns
        -------
        annotation_groups : List[etree.Element]
            list of xmls for annotation groups
        """
        if group_color is None:
            group_color = annotations[0].render_color

        unique_groups = set([a.group_name for a in annotations])
        annotation_groups = []

        for group_name in unique_groups:
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


def load_asap_annotations_from_xml(xml_path):
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
        res = f.read()

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

