from pathlib import Path
from tempfile import NamedTemporaryFile

from .modeling.postprocessing import load_label_info_from_config, predict_on_batch_ROIs, predict_on_WSI
from .utils.annotations import merge_annotations, create_asap_annotation_file


default_label_info = load_label_info_from_config(Path(__file__).parent / "label_info.ini")


def _aggregate_and_format_annotations(annotations):
    merged_annotation = merge_annotations(annotations)
    tmp_file = NamedTemporaryFile()
    create_asap_annotation_file(merged_annotation, tmp_file.name)

    with open(tmp_file.name, "r") as f:
        xml_annotation = f.read()
    tmp_file.close()

    return xml_annotation


def segment_ROI(ROIs,
                upper_left_coords,
                label_info=None,
                batch_size=64,
                return_masks=False):
    """
    Produce segmentation results on unseen ROIs and return annotations encoded in ASAP format
    Parameters
    ----------
    ROIs: List[PIL.Image]
        list of region-of-interest, encoded in PIL Image format
    upper_left_coords: List[Tuple]
        list of upper left coordinates for every ROI, i.e., (minx, miny) on level 0.
    label_info: pd.DataFrame
        class labels and other attributes, use default_label_info as default
    batch_size: int
        batch size for constructing in memory dataloader
    return_masks: bool
        if True, will also return a list of masks for ROIs. See below for details.

    Returns
    -------
    xml_annotation: str
        a string encoding merged ASAP annotation of these ROIs

    predicted_masks: List[List[numpy.array]], optional
        a nested list containing predicted masks for every ROI;
        every item is a list containing masks for every label class,
        ordered in the same order as labels in label_info dataframe.

    """
    if label_info is None:
        label_info = default_label_info
    predicted_masks, annotations = predict_on_batch_ROIs(ROIs, upper_left_coords, label_info, batch_size)
    xml_annotation = _aggregate_and_format_annotations(annotations)

    return xml_annotation if not return_masks else (xml_annotation, predicted_masks)


def segment_WSI(slide_path,
                label_info=None,
                batch_size=64,
                return_masks=False):
    """
    Produce segmentation results on one unseen slide and return annotations encoded in ASAP format

    Parameters
    ----------
    slide_path: str
        path to the unseen tif or svs format slide
    label_info: pd.DataFrame
        class labels and other attributes, use default_label_info as default
    batch_size: int
        batch size for constructing in memory dataloader
    return_masks: bool
        if True, will also return a list of masks for ROIs. See below for details.

    Returns
    -------
    xml_annotation: str
        a string encoding merged ASAP annotation of these ROIs
    predicted_masks: List[List[numpy.array]], optional
        a nested list containing predicted masks for this slide;
        every item is a list containing masks for every label class,
        ordered in the same order as labels in label_info dataframe.
    """

    if label_info is None:
        label_info = default_label_info

    predicted_masks, annotations = predict_on_WSI(slide_path, label_info, batch_size)
    xml_annotation = _aggregate_and_format_annotations(annotations)

    return xml_annotation if not return_masks else (xml_annotation, predicted_masks)
