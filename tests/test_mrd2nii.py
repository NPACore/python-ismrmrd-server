import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest
from helpers import checkers, write_example

import mrd2nii


def test_read_mrd(tmp_path):
    """Test reading MRD data from file."""
    # Create test MRD file
    data = checkers(4, 4, 4)
    examples = write_example(data, tmp_path)
    mrd_file = examples["mrd"]

    # Test the read_mrd function
    images, headers, meta = mrd2nii.read_mrd(str(mrd_file), "dataset")

    # Verify we got data back
    assert len(images) > 0, "Should read at least one image"
    assert len(headers) == len(images), "Headers should match image count"
    assert len(meta) == len(images), "Meta should match image count"

    # Check image properties
    first_img = images[0]
    assert first_img.ndim == 4, "Image should be 4D (cha, z, y, x)"
    assert first_img.shape[0] == 1, "Should have single channel"
    assert first_img.shape[2] == 4, "Should have 4 rows from test data"
    assert first_img.shape[3] == 4, "Should have 4 columns from test data"


def test_get_sort_key():
    """Test the sorting key function for images."""
    # Create mock data structures
    mock_img = np.zeros((1, 4, 4, 4))
    mock_header = {}

    # Create mock meta with image_index
    class MockMeta:
        def __init__(self, image_index, slice_num=None):
            self.image_index = image_index
            self.slice = slice_num

    # Test sorting by image index
    meta1 = MockMeta(5)
    meta2 = MockMeta(2)

    key1 = mrd2nii.get_sort_key((mock_img, mock_header, meta1))
    key2 = mrd2nii.get_sort_key((mock_img, mock_header, meta2))

    assert key1 == 5, "Should return image_index when slice not available"
    assert key2 == 2, "Should return image_index when slice not available"

    # Test sorting by slice when available
    meta3 = MockMeta(10, 3)
    key3 = mrd2nii.get_sort_key((mock_img, mock_header, meta3))
    assert key3 == 3, "Should return slice when available"


def test_combine_matrix():
    """Test combining image matrices into volume data."""
    # Create test image data - single channel, single slice, 4x4
    img1 = np.ones((1, 1, 4, 4), dtype=np.uint16) * 100
    img2 = np.ones((1, 1, 4, 4), dtype=np.uint16) * 200
    img3 = np.ones((1, 1, 4, 4), dtype=np.uint16) * 300

    image_list = [img1, img2, img3]

    # Test combining multiple single-slice images
    volume = mrd2nii.combine_matrix(image_list)

    # Check volume properties
    assert volume.shape == (4, 4, 3), "Should create 4x4x3 volume from 3 images"
    assert volume.dtype == img1.dtype, "Should preserve data type"

    # Check that data is stacked correctly
    assert np.all(volume[:, :, 0] == 100), "First slice should have value 100"
    assert np.all(volume[:, :, 1] == 200), "Second slice should have value 200"
    assert np.all(volume[:, :, 2] == 300), "Third slice should have value 300"


def test_combine_matrix_multi_slice():
    """Test combining matrix with multi-slice input."""
    # Create single image with multiple slices
    img = np.zeros((1, 3, 4, 4), dtype=np.uint16)
    img[0, 0, :, :] = 100  # First slice
    img[0, 1, :, :] = 200  # Second slice
    img[0, 2, :, :] = 300  # Third slice

    image_list = [img]

    # Test combining single multi-slice image
    volume = mrd2nii.combine_matrix(image_list)

    # Check volume properties
    assert volume.shape == (
        4,
        4,
        3,
    ), "Should create 4x4x3 volume from multi-slice image"

    # Check that data is arranged correctly (note: transpose happens)
    assert np.all(volume[:, :, 0] == 100), "Should preserve slice data"
    assert np.all(volume[:, :, 1] == 200), "Should preserve slice data"
    assert np.all(volume[:, :, 2] == 300), "Should preserve slice data"


def test_get_affine():
    """Test affine matrix calculation."""
    # Create mock data
    mock_img = np.zeros((1, 4, 4, 4))
    mock_header = {}

    # Create mock meta with required attributes
    class MockMeta:
        def __init__(self):
            self.field_of_view = [1.0, 1.0, 0.8]  # mm
            self.read_dir = [1.0, 0.0, 0.0]
            self.phase_dir = [0.0, 1.0, 0.0]
            self.position = [0.0, 0.0, 0.0]
            self.image_index = 0
            self.slice = None

    meta = MockMeta()
    combined_data = [(mock_img, mock_header, meta)]

    # Test affine calculation
    affine = mrd2nii.get_affine(combined_data)

    # Check affine properties
    assert affine.shape == (4, 4), "Affine should be 4x4 matrix"
    assert affine[0, 0] == 0.25, "X voxel spacing should be 1.0/4 = 0.25"
    assert affine[1, 1] == 0.25, "Y voxel spacing should be 1.0/4 = 0.25"
    assert affine[2, 2] == 0.2, "Z voxel spacing should be 0.8/4 = 0.2"
    assert affine[3, 3] == 1.0, "Affine should have 1 in bottom right"


def test_make_nii(tmp_path):
    """Test creating NIfTI image from combined data."""
    # Create test data
    data = checkers(4, 4, 4)
    examples = write_example(data, tmp_path)
    mrd_file = examples["mrd"]

    # Read MRD data
    images, headers, meta = mrd2nii.read_mrd(str(mrd_file), "dataset")
    combined_data = list(zip(images, headers, meta))

    # Test make_nii function
    nii_img = mrd2nii.make_nii(combined_data)

    # Verify NIfTI image properties
    assert isinstance(nii_img, nib.Nifti1Image), "Should create NIfTI image"
    assert nii_img.get_fdata().ndim == 3, "Should have 3D data"
    assert nii_img.header.get_xyzt_units() == ("mm", "sec"), "Should have correct units"

    # Check that data is not empty
    nii_data = nii_img.get_fdata()
    npt.assert_array_equal(data, nii_data)
