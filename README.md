# Lightroom Presets to LUTs Converter

This tool converts Adobe Lightroom presets saved as XMP files into LUTs (Look-Up Tables) in the CUBE format that can be used in video editing software like Adobe Premiere Pro.

## Purpose

LUTs are widely used in video color grading but creating them often requires expensive software. This project aims to provide a free alternative by converting your existing Lightroom presets into LUTs.

## Requirements

- Python 3.x
- Libraries: numpy, opencv-python, xml.etree.ElementTree

## Installation

```bash
pip install numpy opencv-python
```

## Usage

1. Place your Lightroom preset XMP files in the same directory as the script.
2. Run the conversion script:
   ```bash
   python xmp_to_lut.py
   ```
3. The script will generate LUT files (.cube) based on your XMP presets.

## Features

- Converts Lightroom XMP preset files to CUBE format LUTs
- Applies exposure, contrast, highlights, shadows and other adjustments
- Can blend the generated LUT with a reference LUT for color matching

## Notes

- Some Lightroom adjustments might not translate perfectly to LUTs due to differences in how adjustments are applied to still images vs video.
- Test the generated LUTs in your video editing software and provide feedback for improvements.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Any contributions to improve the conversion accuracy are welcome!

## License

MIT
