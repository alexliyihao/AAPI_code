from lxml import etree


def pretty_format_xml(xml_node):
    return etree.tostring(xml_node, pretty_print=True).decode('UTF-8').strip()


def write_xml_to_file(filename, xml_node):
    tree = etree.ElementTree(xml_node)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding="utf-8")


class ASAPAnnotation:
    """
    Custom class for describing annotations
    """
    def __init__(self, geometry, annotation_name, group_name, render_color="#F4FA58"):
        """
        docstring
        """
        self.geometry = geometry
        self.annotation_name = str(annotation_name)
        self.group_name = str(group_name)
        self.render_color = str(render_color)

    def to_xml(self):
        coordinates = self.geometry.exterior.coords[:-1]
        coordinates_attrs = [{"Order": str(i), "X": str(x), "Y": str(y)} for i, (x, y) in enumerate(coordinates)]

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

    

