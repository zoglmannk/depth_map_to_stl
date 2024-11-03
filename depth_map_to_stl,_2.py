# -*- coding: utf-8 -*-
"""Depth map to STL, 2.ipynb

Depth Map to STL

Uses numpy-stl to create files

Each square of four pixels is divided into two triangles

Processing time is ~2min per megapixel

Project is at https://github.com/BillFSmith/depth_map_to_stl

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Map to STL Converter

Converts a depth map image to an STL file for 3D printing.
Each square of four pixels is divided into two triangles.
Processing time is ~2min per megapixel.
"""

import argparse
import sys
import numpy as np
from stl import mesh
import cv2

def validate_ratio(value):
    """Validate the height/width ratio is a positive float."""
    try:
        float_value = float(value)
        if float_value <= 0:
            raise ValueError
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a positive float value")

def load_depth_map(image_path):
    """Load and validate the depth map image."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        sys.exit(1)

def create_mesh_from_depth_map(depth_map, height_width_ratio):
    """Convert depth map to 3D mesh."""
    # Convert image to numpy array and rotate
    im_array = np.array(depth_map)
    im_array = np.rot90(im_array, -1, (0,1))
    
    mesh_size = [im_array.shape[0], im_array.shape[1]]
    mesh_max = np.max(im_array)
    
    # Handle both grayscale and color images
    if len(im_array.shape) == 3:
        scaled_mesh = mesh_size[0] * height_width_ratio * im_array[:,:,0] / mesh_max
    else:
        scaled_mesh = mesh_size[0] * height_width_ratio * im_array / mesh_max
    
    # Create mesh
    mesh_shape = mesh.Mesh(np.zeros((mesh_size[0] - 1) * (mesh_size[1] - 1) * 2, 
                                  dtype=mesh.Mesh.dtype))
    
    # Generate triangles
    for i in range(0, mesh_size[0]-1):
        for j in range(0, mesh_size[1]-1):
            mesh_num = i * (mesh_size[1]-1) + j
            
            # First triangle
            mesh_shape.vectors[2 * mesh_num][2] = [i, j, scaled_mesh[i,j]]
            mesh_shape.vectors[2 * mesh_num][1] = [i, j+1, scaled_mesh[i,j+1]]
            mesh_shape.vectors[2 * mesh_num][0] = [i+1, j, scaled_mesh[i+1,j]]
            
            # Second triangle
            mesh_shape.vectors[2 * mesh_num + 1][0] = [i+1, j+1, scaled_mesh[i+1,j+1]]
            mesh_shape.vectors[2 * mesh_num + 1][1] = [i, j+1, scaled_mesh[i,j+1]]
            mesh_shape.vectors[2 * mesh_num + 1][2] = [i+1, j, scaled_mesh[i+1,j]]
    
    return mesh_shape

def main():
    parser = argparse.ArgumentParser(
        description='Convert a depth map image to an STL file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s -i input.png -o output.stl -r 0.5

Note: The ratio parameter represents the desired height/width ratio of the final model.
      For example, if your object should be 10cm tall and 20cm wide, use 0.5
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input depth map image file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output STL file')
    parser.add_argument('-r', '--ratio', type=validate_ratio, required=True,
                        help='Desired height/width ratio of the final model')
    
    args = parser.parse_args()
    
    # Load the depth map
    depth_map = load_depth_map(args.input)
    
    # Create the mesh
    mesh_shape = create_mesh_from_depth_map(depth_map, args.ratio)
    
    # Save the result
    try:
        mesh_shape.save(args.output)
        print(f"Successfully created STL file: {args.output}")
    except Exception as e:
        print(f"Error saving STL file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
