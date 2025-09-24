# This script updates the EXIF GPS coordinates of image files
# based on data from a tab-delimited text file.
# It can be run from a standard Windows Command Prompt.

import os
import csv
import argparse
import piexif

# --- Helper function to convert decimal degrees to DMS format for EXIF ---
def decimal_to_dms(decimal_degree):
    """
    Converts a decimal degree coordinate to the required EXIF DMS format.
    The format is a list of tuples: [(degree, 1), (minute, 1), (second, 1)]
    """
    is_positive = decimal_degree >= 0
    decimal_degree = abs(decimal_degree)

    degrees = int(decimal_degree)
    minutes_decimal = (decimal_degree - degrees) * 60
    minutes = int(minutes_decimal)
    seconds_decimal = (minutes_decimal - minutes) * 60

    # EXIF stores fractions as a numerator and denominator
    # We will use 10000 to maintain some precision for seconds
    seconds_numerator = int(seconds_decimal * 10000)
    seconds_denominator = 10000

    return [(degrees, 1), (minutes, 1), (seconds_numerator, seconds_denominator)]

# --- Main script logic ---
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Updates image EXIF GPS data from a tab-delimited file.")
    parser.add_argument("tab_file", help="Path to the tab-delimited file containing image data.")
    parser.add_argument("image_folder", help="Path to the folder containing the images.")
    args = parser.parse_args()

    # Get file paths from arguments
    tab_file_path = args.tab_file
    image_folder_path = args.image_folder

    # Check if the provided paths are valid
    if not os.path.exists(tab_file_path):
        print(f"Error: Tab-delimited file not found at '{tab_file_path}'")
        return
    if not os.path.isdir(image_folder_path):
        print(f"Error: Image folder not found at '{image_folder_path}'")
        return

    # --- Initialize counters for the summary report ---
    successful_updates = 0
    failed_updates = 0
    
    print("Processing images from tab-delimited file...")
    
    # Open and read the tab-delimited file
    try:
        # We specify the delimiter as a tab character '\t'
        with open(tab_file_path, 'r', newline='') as tabfile:
            reader = csv.DictReader(tabfile, delimiter='\t')
            rows = list(reader)  # Read all rows into a list to get the total count
            total_items = len(rows)

            # --- UPDATED: These variables now match your column headers ---
            file_name_col = "Path"
            x_coord_col = "X"  # Longitude
            y_coord_col = "Y"  # Latitude
            z_coord_col = "Z"  # Elevation

            for row in rows:
                try:
                    file_name = row[file_name_col]
                    longitude = float(row[x_coord_col])
                    latitude = float(row[y_coord_col])
                    elevation = float(row[z_coord_col])

                    image_path = os.path.join(image_folder_path, file_name)

                    if not os.path.exists(image_path):
                        print(f"Warning: Image not found: {image_path}. Skipping.")
                        failed_updates += 1
                        continue

                    # Load existing EXIF data
                    exif_dict = piexif.load(image_path)

                    # Convert coordinates to DMS format
                    lat_dms = decimal_to_dms(latitude)
                    lon_dms = decimal_to_dms(longitude)
                    
                    # Determine latitude and longitude references (N/S, E/W)
                    lat_ref = "N" if latitude >= 0 else "S"
                    lon_ref = "E" if longitude >= 0 else "W"
                    
                    # Determine altitude reference (above/below sea level) and format for EXIF
                    # EXIF altitude is a tuple (numerator, denominator). We'll use a denominator of 100
                    # for good precision.
                    if elevation >= 0:
                        altitude_ref = 0  # Above sea level
                    else:
                        altitude_ref = 1  # Below sea level
                    altitude_tuple = (int(abs(elevation) * 100), 100)


                    # Update the GPSInfo dictionary
                    if "GPS" not in exif_dict:
                        exif_dict["GPS"] = {}

                    exif_dict["GPS"][piexif.GPSIFD.GPSVersionID] = (2, 0, 0, 0)
                    exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref
                    exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = lat_dms
                    exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref
                    exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = lon_dms
                    exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = altitude_ref
                    exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = altitude_tuple

                    # Insert the modified EXIF data back into the image file
                    exif_bytes = piexif.dump(exif_dict)
                    piexif.insert(exif_bytes, image_path)

                    print(f"Successfully updated EXIF for: {image_path}")
                    successful_updates += 1

                except (KeyError, ValueError) as e:
                    print(f"Error processing row for '{row.get(file_name_col, 'unknown')}': {e}. Skipping row.")
                    failed_updates += 1
                except Exception as e:
                    print(f"An unexpected error occurred with image {file_name}: {e}")
                    failed_updates += 1

    except FileNotFoundError:
        print(f"Error: The tab-delimited file was not found at '{tab_file_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # --- Print summary report ---
    print("\n--- Processing Summary ---")
    print(f"Total items in file: {total_items}")
    print(f"Successfully updated: {successful_updates}")
    print(f"Failed updates: {failed_updates}")
    print("--------------------------")

    print("\nScript completed.")

if __name__ == "__main__":
    main()
