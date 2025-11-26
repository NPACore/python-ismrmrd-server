#!/usr/bin/python3

import argparse
import os
import re

import ismrmrd
import nibabel as nib
import numpy as np


def read_mrd(filename, in_group):
    """
    :param filename: path to mrd h5 file
    :param in_group: group of h5 to read
    :returns: (image,ImageHeader, Meta)
    """
    header = []
    meta = []
    image = []

    dset = ismrmrd.Dataset(filename, in_group, False)
    groups = dset.list()
    # Collect all image data
    image_groups = [name for name in groups if name.startswith("image")]
    for grp in image_groups:
        for num in range(dset.number_of_images(grp)):
            this_img = dset.read_image(grp, num)
            image.append(this_img.data)
            header.append(ismrmrd.Meta.deserialize(this_img.attribute_string))
            meta.append(this_img.getHead())

    return (image, header, meta)


def get_sort_key(img_header_tuple):
    "Sort by slice or image_index"
    img, header, meta = img_header_tuple
    # Try to use slice position, fall back to image index
    if hasattr(meta, "slice") and meta.slice is not None:
        return meta.slice
    return meta.image_index


def combine_matrix(image_list):
    # Determine volume dimensions
    first_img = image_list[0]
    print(first_img.shape)
    time, slices, rows, cols = first_img.shape

    # Handle different data organization
    if len(image_list) == 1 and slices > 1:
        # Single image with multiple slices in z dimension
        volume_data = np.squeeze(first_img, axis=0)  # Remove channel dimension
        volume_data = np.transpose(volume_data, (2, 1, 0))  # Reorder to (x, y, z)
    else:
        # Multiple separate images - stack them
        slices = len(image_list)
        volume_data = np.zeros((rows, cols, slices), dtype=first_img.dtype)

        for i, img in enumerate(image_list):
            if img.shape[1] != 1:
                print(
                    f"Warning: Image {i} has {img.shape[1]} slices, using first slice only"
                )
            # Extract single slice: [cha, z, y, x] -> [y, x]
            slice_data = np.squeeze(img[:, 0, :, :])
            volume_data[:, :, i] = slice_data

        # Transpose to NIfTI convention (x, y, z) - note: nibabel expects (i, j, k)
        volume_data = np.transpose(volume_data, (1, 0, 2))
    return volume_data


def get_affine(combined_data):
    first_img = combined_data[0][0]
    hdr = combined_data[0][2]  # meta is at index 2
    time, slices, rows, cols = first_img.shape

    # Calculate voxel sizes from field of view
    pixelspacing_x = hdr.field_of_view[0] / cols
    pixelspacing_y = hdr.field_of_view[1] / rows

    if len(combined_data) > 1:
        try:
            # Use slice spacing between first two images if available
            pos1 = combined_data[0][2].position
            pos2 = combined_data[1][2].position
            slice_spacing = np.linalg.norm(np.array(pos2) - np.array(pos1))
        except:
            slice_spacing = hdr.field_of_view[2]
    else:
        # Single image - use field of view directly
        if slices > 1:
            # Multi-slice image - divide field of view by number of slices
            slice_spacing = hdr.field_of_view[2] / slices
        else:
            # Single slice - use full field of view as slice thickness
            slice_spacing = hdr.field_of_view[2]

    # Ensure slice spacing is not zero or negative
    if slice_spacing <= 0:
        print(f"Warning: Invalid slice spacing {slice_spacing}, using default 1.0mm")
        slice_spacing = 1.0

    # Create affine transformation matrix
    # Default to identity with proper voxel spacing
    affine = np.eye(4)
    affine[0, 0] = pixelspacing_x
    affine[1, 1] = pixelspacing_y
    affine[2, 2] = slice_spacing

    # Try to set orientation from MRD header if available
    try:
        # Get image orientation from first image
        read_dir = hdr.read_dir
        phase_dir = hdr.phase_dir
        slice_dir = np.cross(read_dir, phase_dir)

        # Set orientation in affine matrix
        affine[0:3, 0] = read_dir * pixelspacing_x
        affine[1:3, 1] = phase_dir * pixelspacing_y
        affine[0:3, 2] = slice_dir * slice_spacing

        # Set position
        if hasattr(hdr, "position"):
            affine[0:3, 3] = hdr.position
    except:
        print("Warning: Could not set proper orientation, using default")

    return affine


def make_nii(combined_data):
    combined_data.sort(key=get_sort_key)
    volume_data = combine_matrix([img for (img, h, m) in combined_data])
    affine = get_affine(combined_data)

    # Create NIfTI image
    nii_img = nib.Nifti1Image(volume_data, affine)

    # Set some basic header information
    nii_img.header.set_xyzt_units("mm", "sec")

    print(f"Volume dimensions: {volume_data.shape}")
    print(
        f"Voxel spacing: {affine[0,0]:.3f} x {affine[1,1]:.3f} x {affine[2,2]:.3f} mm"
    )
    return nii_img


def main(args):
    images, headers, meta = read_mrd(args.filename, args.in_group)

    if len(images) == 0:
        print("No suitable images found for NIfTI conversion")
        return

    if not args.out_file:
        args.out_file = re.sub(".h5$", ".nii.gz", args.filename)
        print("Output file not specified -- using %s" % args.out_file)

    combined_data = list(zip(images, headers, meta))

    nii_img = make_nii(combined_data)

    # nii_img.header['pixdim'][4] = 1.0  # TR in seconds - would need sequence info

    # Save NIfTI file
    print(f"Writing NIfTI file: {args.out_file}")

    nib.save(nii_img, args.out_file)
    print(f"Successfully wrote {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MRD image file to NIfTI format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input MRD file")
    parser.add_argument("-g", "--in-group", help="Input data group")
    parser.add_argument("-o", "--out-file", help="Output NIfTI file (.nii.gz)")

    args = parser.parse_args()

    main(args)
