from pathlib import Path
from tempfile import NamedTemporaryFile

from .modeling.postprocessing import load_label_info_from_config, predict_on_ROI, predict_on_WSI
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

    if label_info is None:
        label_info = default_label_info
    predicted_masks, annotations = predict_on_ROI(ROIs, upper_left_coords, label_info, batch_size)
    xml_annotation = _aggregate_and_format_annotations(annotations)

    return xml_annotation if not return_masks else (xml_annotation, predicted_masks)


def segment_WSI(slide_path,
                label_info=None,
                batch_size=64,
                return_masks=False):

    if label_info is None:
        label_info = default_label_info

    predicted_masks, annotations = predict_on_WSI(slide_path, label_info, batch_size)
    xml_annotation = _aggregate_and_format_annotations(annotations)

    return xml_annotation if not return_masks else (xml_annotation, predicted_masks)
