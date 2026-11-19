import json
import warnings

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from xraymind.dicom import dicom_to_array, dicom_to_png, redact_dicom, write_safe_metadata_json
from xraymind.preprocessing import prepare_xray_tensor


def make_dicom(path, photometric='MONOCHROME2'):
    metadata = FileMetaDataset()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian
    metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=metadata, preamble=b'\0' * 128)
    dataset.SOPClassUID = metadata.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dataset.PatientName = 'Example^Patient'
    dataset.PatientID = 'test-id'
    dataset.PatientBirthDate = '19800101'
    dataset.PatientAge = '046Y'
    dataset.PatientSex = 'F'
    dataset.Modality = 'DX'
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.array([[0, 100], [200, 300]], dtype='<u2').tobytes()
    dataset.add_new((0x0011, 0x0010), 'LO', 'PRIVATE DATA')
    dataset.save_as(path, enforce_file_format=True)
    return dataset


def test_dicom_conversion_metadata_and_tensor(tmp_path):
    path = tmp_path / 'image.dcm'
    make_dicom(path)
    arr = dicom_to_array(path)
    assert arr.shape == (2, 2) and arr.dtype == np.uint8
    assert arr[0, 0] == 0 and arr[1, 1] >= 254
    assert dicom_to_png(path, tmp_path / 'preview.png').exists()
    tensor = prepare_xray_tensor(path)
    assert tuple(tensor.shape) == (1, 1, 224, 224)
    metadata = json.loads(write_safe_metadata_json(path, tmp_path / 'metadata.json').read_text())
    assert metadata['Modality'] == 'DX'
    assert 'PatientName' not in metadata and 'PatientID' not in metadata


def test_monochrome_one_is_inverted(tmp_path):
    path = tmp_path / 'image.dcm'
    make_dicom(path, 'MONOCHROME1')
    arr = dicom_to_array(path)
    assert arr[0, 0] == 255 and arr[1, 1] <= 1


def test_redaction_preserves_valid_dicom_value_types(tmp_path):
    path = tmp_path / 'image.dcm'
    original = make_dicom(path)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        redacted = redact_dicom(path, tmp_path / 'redacted.dcm')
    result = pydicom.dcmread(redacted)
    assert str(result.PatientName) == 'REDACTED'
    assert result.PatientID == 'REDACTED'
    assert result.PatientBirthDate == ''
    assert result.PatientAge == ''
    assert result.PatientSex == ''
    assert not any(element.tag.is_private for element in result)
    assert result.PixelData == original.PixelData


@pytest.mark.parametrize('shape', [(16, 24), (24, 16), (16, 24, 3)])
def test_preprocessing_non_square_images(shape):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    tensor = prepare_xray_tensor(image)
    assert tuple(tensor.shape) == (1, 1, 224, 224)
    assert np.isfinite(tensor.numpy()).all()


def test_overlay_uses_same_center_crop_as_model(tmp_path, monkeypatch):
    import torchxrayvision as xrv
    from matplotlib.axes import Axes
    from xraymind import visualization

    image = np.arange(24, dtype=np.uint8).reshape(4, 6)
    crop = xrv.datasets.XRayCenterCrop()(image[None, ...])[0]
    expected = visualization._to_grayscale_uint8(crop, size=(8, 8))
    displayed = []
    original_imshow = Axes.imshow
    def capture(self, array, *args, **kwargs):
        displayed.append(np.asarray(array))
        return original_imshow(self, array, *args, **kwargs)
    monkeypatch.setattr(Axes, 'imshow', capture)
    visualization.save_heatmap_overlay(image, np.ones((4, 4)), tmp_path / 'overlay.png', size=8)
    np.testing.assert_array_equal(displayed[0], expected)
