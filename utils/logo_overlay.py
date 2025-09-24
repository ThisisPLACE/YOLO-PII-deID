# This script projects a flat logo onto the bottom of a 360-degree
# equirectangular image, correcting for the distortion that makes the
# logo appear conical.

# Required library: Pillow (PIL). Install it with: pip install Pillow
import os
import argparse
from PIL import Image

def add_projected_logo(image_path, logo_path, output_path):
    """
    Projects a flat logo onto the bottom of a 360-degree equirectangular image
    to correct for geometric distortion. The logo will appear rectangular in
    a 360-degree viewer.

    Args:
        image_path (str): The path to the input 360-degree image file.
        logo_path (str): The path to the logo image file (e.g., a PNG with transparency).
        output_path (str): The path to save the output image.
    """
    try:
        # Open the main 360-degree image and get its original metadata
        original_image = Image.open(image_path)
        
        # We need to get the EXIF data before converting the image to RGBA
        exif_data = original_image.info.get('exif')

        # Convert to RGBA for consistency in handling transparency
        main_image = original_image.convert("RGBA")
        main_image_width, main_image_height = main_image.size

        # Open the logo image. This is assumed to be a PNG with transparency.
        logo = Image.open(logo_path).convert("RGBA")

        # --- Adjustable Parameters ---
        # These variables control the logo's appearance and position on the projected plane.
        # Adjust these to change the logo's size and placement.
        logo_width_scale = 1  # Controls the width of the logo relative to the image width.
        logo_height_scale = 0.15 # Controls the height of the logo relative to the image height.
        
        # New variable to determine the horizontal position of the logo using degrees.
        # 0 degrees is the far left, 180 degrees is the center, and 360 degrees is the far right.
        # The value is mapped to the image's width.
        logo_heading_degrees = 0
        
        # Position of the logo on the projected plane
        # 0.0 is at the very bottom seam, higher values move it up
        logo_y_offset = 0

        # Convert the heading in degrees to a proportional offset value (0.0 to 1.0)
        logo_x_offset = logo_heading_degrees / 360.0

        # Calculate the size of the projected logo area.
        # This is a key step to ensure the logo appears rectangular.
        projected_logo_width = int(main_image_width * logo_width_scale)
        projected_logo_height = int(main_image_height * logo_height_scale)

        # Calculate the position of the logo on the projected plane
        x_position = int((main_image_width - projected_logo_width) * logo_x_offset)
        y_position = int(main_image_height * (1.0 - logo_y_offset) - projected_logo_height)

        # Resize the logo to fit the projected area, maintaining aspect ratio
        resized_logo = logo.resize((projected_logo_width, projected_logo_height), Image.Resampling.LANCZOS)
        
        # Paste the resized logo onto the main image
        main_image.paste(resized_logo, (x_position, y_position), resized_logo)
        
        # Save the new image with the logo and the original metadata
        try:
            _, ext = os.path.splitext(output_path)
            if not ext or ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                raise ValueError("Output path is missing a valid file extension (e.g., .jpg, .png).")

            if main_image.mode == 'RGBA' and ext.lower() in ['.jpg', '.jpeg']:
                # For JPEGs, convert from RGBA to RGB as they don't support transparency
                rgb_image = Image.new('RGB', main_image.size, (255, 255, 255))
                rgb_image.paste(main_image, (0, 0), main_image)
                
                # Save with high quality and disabled subsampling
                if exif_data:
                    rgb_image.save(output_path, exif=exif_data, quality=95, subsampling=0)
                else:
                    rgb_image.save(output_path, quality=95, subsampling=0)
                print(f"Saved (converted to RGB): {output_path}")
            else:
                # Save with high quality and disabled subsampling for other formats if applicable
                if exif_data:
                    main_image.save(output_path, exif=exif_data, quality=95, subsampling=0)
                else:
                    main_image.save(output_path, quality=95, subsampling=0)
                print(f"Saved: {output_path}")
        except ValueError as ve:
            print(f"Error saving {output_path}: {ve}")
        except Exception as save_e:
            print(f"Error saving {output_path}: {save_e}")

    except FileNotFoundError:
        print(f"Error: One of the files was not found. Please check that '{image_path}' and '{logo_path}' exist.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {image_path}: {e}")

def batch_project_logos(input_folder, logo_path, output_folder):
    """
    Projects a logo onto all images in a folder and saves them to a new folder.

    Args:
        input_folder (str): Path to the folder containing input images.
        logo_path (str): Path to the logo image file.
        output_folder (str): Path to the folder to save the output images.
    """
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'.")
        return
    if not os.path.isfile(logo_path):
        print(f"Error: Logo file not found at '{logo_path}'.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")
    
    print(f"\nStarting batch process for images in '{input_folder}'...")

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is a supported image type
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Call the main function to add the logo
            add_projected_logo(image_path, logo_path, output_path)

    print("\nBatch process finished.")


# --- Main script logic to handle command-line arguments ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch projects a logo onto all 360-degree images in a folder.")
    parser.add_argument("input_folder", help="Path to the folder containing input 360-degree images.")
    parser.add_argument("logo_path", help="Path to the logo image file (e.g., a PNG with transparency).")
    parser.add_argument("output_folder", help="Path to the folder where the new images will be saved.")

    args = parser.parse_args()

    # Call the batch processing function with the provided arguments
    batch_project_logos(args.input_folder, args.logo_path, args.output_folder)
