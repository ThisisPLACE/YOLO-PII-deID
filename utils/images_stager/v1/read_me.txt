**Features:**
- **Finds stitched folders**: Recursively searches the parent directory for folders containing "stitched" in the path
- **Identifies images**: Recognizes .jpg, .jpeg, .png, .bmp, .gif, .tiff, and .webp formats
- **Preserves structure**: Maintains the same directory hierarchy in the output folder
- **Compresses images**: Reduces resolution to 50% (half width and height)
- **Reduces quality**: Saves as JPG with 50% quality
- **Strips metadata**: Removes EXIF data from images
- **Robust error handling**: Handles various image formats and reports progress

**Usage:**
```bash
python image_stitched_processor.py /path/to/parent/directory /path/to/output/directory
```

**Example:**
```bash
python image_stitched_processor.py ~/photos ~/processed_photos
```

The script will scan through all folders, find those with "stitched" in the path, and process all images within them, creating a matching directory structure in your output folder with compressed images.