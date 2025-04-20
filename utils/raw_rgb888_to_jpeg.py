import numpy as np
from PIL import Image
import os

def raw_to_jpeg(raw_path, output_path=None, width=None, height=None):
    """
    Convert a RAW RGB888 file to JPEG
    
    Args:
        raw_path (str): Path to the .raw file
        output_path (str): Path for output JPEG (if None, uses same name as raw)
        width (int): Width of the image (if None, tries to detect from filename)
        height (int): Height of the image (if None, tries to detect from filename)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # If dimensions not provided, try to detect from filename
        if width is None or height is None:
            if "resized_" in raw_path:
                # Resized images are 192x192
                width, height = 192, 192
            elif "new3_rgb888_" in raw_path:
                # Full images are 800x600
                width, height = 800, 600
            elif "face_" in raw_path:
                # Parse dimensions from filename (e.g., "face_168x166_0.raw")
                try:
                    dims = raw_path.split('_')[1]  # Get "168x166" part
                    width, height = map(int, dims.split('x'))
                    print(f"Parsed dimensions from filename: {width}x{height}")
                except:
                    print("Could not parse dimensions from filename")
                    return False
            else:
                # For face crops, we need to determine from file size
                file_size = os.path.getsize(raw_path)
                pixels = file_size // 3  # 3 bytes per pixel (RGB888)
                print(f"File size: {file_size} bytes")
                print(f"Total pixels: {pixels}")
                
                # Try common image dimensions that could match the pixel count
                common_dimensions = [
                    (168, 166),  # 27,888 pixels
                    (166, 168),  # 27,888 pixels
                    (192, 144),  # 27,648 pixels
                    (144, 192),  # 27,648 pixels
                    (160, 175),  # 28,000 pixels
                ]
                
                for w, h in common_dimensions:
                    if abs(w * h - pixels) < 100:  # Allow small tolerance
                        width, height = w, h
                        print(f"Found matching dimensions: {width}x{height}")
                        break
                else:
                    # If no common dimensions match, default to square
                    width = height = int(np.ceil(np.sqrt(pixels)))
                    print(f"Using square dimensions: {width}x{width}")
        
        # Read the raw file
        with open(raw_path, 'rb') as f:
            raw_data = f.read()
        
        # Convert to numpy array
        img_array = np.frombuffer(raw_data, dtype=np.uint8)
        
        # Reshape to image dimensions
        img_array = img_array.reshape((height, width, 3))
        
        # Create PIL Image
        img = Image.fromarray(img_array, 'RGB')
        
        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.splitext(raw_path)[0] + '.jpg'
        
        # Save as JPEG
        img.save(output_path, 'JPEG')
        print(f"Converted {raw_path} to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {raw_path}: {e}")
        return False

# Example usage:
if __name__ == "__main__":
    # Convert all resized images in the folder
    import glob
    resized_files = glob.glob("/Volumes/NO NAME/new*_rgb888_*.raw")
    for resized_file in resized_files:
        raw_to_jpeg(resized_file)  # Assuming 192x192 dimensions for resized images
    
    # Convert all new face crops in the folder (dimensions determined from file size)
    face_files = glob.glob("/Volumes/NO NAME/face_*.raw")
    for face_file in face_files:
        raw_to_jpeg(face_file)
    # # Convert a face crop (dimensions determined from file size)
    # raw_to_jpeg("face_0.raw")
    
    # # Or specify dimensions manually
    # raw_to_jpeg("face_0.raw", width=100, height=100)  # if you know the dimensions