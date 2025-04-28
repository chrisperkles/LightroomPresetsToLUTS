import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import skimage.exposure
import skimage.color
import sys

def parse_xmp(file_path):
    """Parse Lightroom XMP file and extract adjustment parameters."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/'}
    adjustments = {}
    
    for desc in root.findall('.//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description'):
        print(f"Found Description element with attributes: {desc.attrib}")
        for key, value in desc.attrib.items():
            if 'crs:' in key or key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                param = key.replace('{http://ns.adobe.com/camera-raw-settings/1.0/}', '')
                adjustments[param] = float(value) if value.replace('.', '').replace('-', '').isdigit() else value
                print(f"Parsed adjustment: {param} = {value}")
    
    # Parse tone curves
    tone_curve = adjustments.get('ToneCurvePV2012', None)
    if tone_curve:
        points = [tuple(map(float, p.split(', '))) for p in tone_curve.split(';') if p]
        adjustments['ToneCurvePV2012'] = points
    
    # Parse per-channel tone curves
    for channel in ['Red', 'Green', 'Blue']:
        curve_key = f'ToneCurvePV2012{channel}'
        curve = adjustments.get(curve_key, None)
        if curve:
            points = [tuple(map(float, p.split(', '))) for p in curve.split(';') if p]
            adjustments[curve_key] = points
    
    return adjustments

def generate_hald_clut(size=33):
    """Generate a Hald CLUT image."""
    width = size * size
    height = size
    r, g, b = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), np.linspace(0, 1, size))
    clut = np.stack((r.flatten(), g.flatten(), b.flatten()), axis=1)
    image = clut.reshape((height, width, 3))
    return image

def srgb_to_linear(image):
    """Convert sRGB image to linear RGB."""
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(image):
    """Convert linear RGB to sRGB."""
    return np.where(image <= 0.0031308, 12.92 * image, 1.055 * image ** (1 / 2.4) - 0.055)

def adjust_exposure(image, exposure):
    """Adjust exposure in linear space with increased intensity."""
    exposure = float(exposure)  # Ensure exposure is a float
    adjusted = image * (2 ** (exposure * 1.5))  # Increase exposure effect by 1.5x
    return np.clip(adjusted, 0, 1)

def adjust_contrast(image, contrast):
    """Adjust contrast in linear space with increased intensity."""
    contrast = float(contrast)  # Ensure contrast is a float
    factor = (contrast + 100) / 100 * 1.2  # Increase contrast effect by 1.2x
    mid = 0.5
    image = mid + factor * (image - mid)
    return np.clip(image, 0, 1)

def adjust_tone_curve(image, curve_points):
    """Apply tone curve adjustment in linear space."""
    if not curve_points:
        return image
    x = np.array([p[0] / 255 for p in curve_points])
    y = np.array([p[1] / 255 for p in curve_points])
    lut = np.interp(np.linspace(0, 1, 256), x, y)
    return lut[(image * 255).astype(int)] / 255

def adjust_channel_tone_curve(image, red_curve, green_curve, blue_curve):
    """Apply per-channel tone curve adjustments in linear space."""
    result = image.copy()
    if red_curve:
        result[..., 0] = adjust_tone_curve(result[..., 0], red_curve)
    if green_curve:
        result[..., 1] = adjust_tone_curve(result[..., 1], green_curve)
    if blue_curve:
        result[..., 2] = adjust_tone_curve(result[..., 2], blue_curve)
    return result

def adjust_saturation(image, saturation):
    """Apply saturation adjustment with increased intensity."""
    factor = (saturation + 100) / 100 * 1.3  # Increase saturation effect by 1.3x
    hsv = skimage.color.rgb2hsv(linear_to_srgb(image))
    hsv[..., 1] *= factor
    return srgb_to_linear(skimage.color.hsv2rgb(np.clip(hsv, 0, 1)))

def adjust_hsl(image, hue_adjustments, sat_adjustments, lum_adjustments):
    """Adjust hue, saturation, and luminance for specific color ranges with increased intensity."""
    hsv = skimage.color.rgb2hsv(linear_to_srgb(image))
    
    color_ranges = {
        'Red': (0, 0.083),
        'Orange': (0.083, 0.167),
        'Yellow': (0.167, 0.25),
        'Green': (0.25, 0.5),
        'Aqua': (0.5, 0.583),
        'Blue': (0.583, 0.75),
        'Purple': (0.75, 0.833),
        'Magenta': (0.833, 1.0)
    }
    
    for color, (start, end) in color_ranges.items():
        mask = (hsv[..., 0] >= start) & (hsv[..., 0] < end)
        hue_key = f'HueAdjustment{color}'
        sat_key = f'SaturationAdjustment{color}'
        lum_key = f'LuminanceAdjustment{color}'
        if hue_key in hue_adjustments:
            hue_value = float(hue_adjustments.get(hue_key, 0))
            hsv[mask, 0] += (hue_value / 360.0) * 1.5  # Increase hue effect by 1.5x
        if sat_key in sat_adjustments:
            sat_value = float(sat_adjustments.get(sat_key, 0))
            hsv[mask, 1] *= (1 + sat_value / 100 * 1.5)  # Increase saturation effect by 1.5x
        if lum_key in lum_adjustments:
            lum_value = float(lum_adjustments.get(lum_key, 0))
            hsv[mask, 2] *= (1 + lum_value / 100 * 1.5)  # Increase luminance effect by 1.5x
    
    hsv[..., 0] = (hsv[..., 0] % 1.0)  # Wrap hue around 0-1 range
    hsv = np.clip(hsv, 0, 1)
    return srgb_to_linear(skimage.color.hsv2rgb(hsv))

def adjust_highlights_shadows(image, highlights, shadows, whites, blacks):
    """Adjust highlights, shadows, whites, and blacks with increased intensity."""
    highlights = float(highlights)
    shadows = float(shadows)
    whites = float(whites)
    blacks = float(blacks)
    
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    mask_shadows = luminance < 0.5
    mask_highlights = luminance >= 0.5
    
    factor = (shadows + 100) / 100 * 1.5  # Increase shadows effect by 1.5x
    image[mask_shadows] *= factor
    factor = (highlights + 100) / 100 * 1.5  # Increase highlights effect by 1.5x
    image[mask_highlights] /= max(0.01, factor)
    image = np.clip(image + (whites / 100 * 0.1) - (blacks / 100 * 0.1), 0, 1)  # Increase whites/blacks effect
    return image

def apply_adjustments(image, adjustments):
    """Apply adjustments in sequence in linear RGB space."""
    # Convert input image to linear RGB space
    result = srgb_to_linear(image.copy())
    
    # Basic exposure and contrast
    if 'Exposure2012' in adjustments:
        result = adjust_exposure(result, adjustments['Exposure2012'])
    if 'Contrast2012' in adjustments:
        result = adjust_contrast(result, adjustments['Contrast2012'])
    
    # Highlights, Shadows, Whites, Blacks
    highlights = adjustments.get('Highlights2012', 0)
    shadows = adjustments.get('Shadows2012', 0)
    whites = adjustments.get('Whites2012', 0)
    blacks = adjustments.get('Blacks2012', 0)
    result = adjust_highlights_shadows(result, highlights, shadows, whites, blacks)
    
    # Overall tone curve
    if 'ToneCurvePV2012' in adjustments:
        result = adjust_tone_curve(result, adjustments['ToneCurvePV2012'])
    
    # Per-channel tone curves
    red_curve = adjustments.get('ToneCurvePV2012Red', [])
    green_curve = adjustments.get('ToneCurvePV2012Green', [])
    blue_curve = adjustments.get('ToneCurvePV2012Blue', [])
    result = adjust_channel_tone_curve(result, red_curve, green_curve, blue_curve)
    
    # Overall saturation
    if 'Saturation' in adjustments:
        result = adjust_saturation(result, adjustments['Saturation'])
    
    # Color-specific HSL adjustments
    hue_adjustments = {k: v for k, v in adjustments.items() if k.startswith('HueAdjustment')}
    sat_adjustments = {k: v for k, v in adjustments.items() if k.startswith('SaturationAdjustment')}
    lum_adjustments = {k: v for k, v in adjustments.items() if k.startswith('LuminanceAdjustment')}
    result = adjust_hsl(result, hue_adjustments, sat_adjustments, lum_adjustments)
    
    # Convert back to sRGB space for output
    return linear_to_srgb(np.clip(result, 0, 1))

def load_cube_file(file_path):
    """Load a .cube file into a 3D numpy array."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    size = None
    data = []
    for line in lines:
        line = line.strip()
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[-1])
        elif line and not line.startswith('#') and not line.startswith('TITLE'):
            r, g, b = map(float, line.split())
            data.append([r, g, b])
    
    if size is None:
        raise ValueError("Could not find LUT_3D_SIZE in .cube file")
    
    data = np.array(data).reshape((size, size, size, 3))
    return data

def apply_reference_adjustment(image, reference_lut, adjustments):
    """Apply adjustment based on a reference LUT to better match its output."""
    # Assuming reference_lut is a 4D array with shape (size, size, size, 3)
    # We need to adjust our generated adjustments to blend with this LUT
    print(f"Reference LUT shape: {reference_lut.shape}")
    lut_size = reference_lut.shape[0]
    # Create a grid for the LUT
    r, g, b = np.meshgrid(np.linspace(0, 1, lut_size),
                         np.linspace(0, 1, lut_size),
                         np.linspace(0, 1, lut_size))
    grid = np.stack([r, g, b], axis=-1)
    # Apply the current image adjustments to this grid
    adjusted_grid = apply_adjustments(grid, adjustments)
    # Now blend with reference LUT
    blended_lut = adjusted_grid * 0.3 + reference_lut * 0.7
    return blended_lut

def image_to_cube(image, size, output_file):
    """Convert adjusted Hald CLUT to .cube LUT with increased LUT size to match working example."""
    data = image.reshape((size, size, size, 3))
    with open(output_file, 'w') as f:
        f.write("TITLE \"LUT from XMP\"\n")
        f.write(f"LUT_3D_SIZE {size}\n\n")
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    rgb = data[b, g, r]
                    f.write(f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n")

def xmp_to_lut(xmp_file, output_lut_file, lut_size=64, reference_lut_file=None):
    """Convert XMP file to LUT, optionally using a reference LUT for adjustment."""
    # Parse XMP adjustments
    adjustments = parse_xmp(xmp_file)
    print(f"Parsed adjustments: {adjustments.keys()}")
    
    # Load Hald CLUT identity image
    hald_image = generate_hald_clut(lut_size)
    print(f"Hald image shape: {hald_image.shape}")
    
    # Apply adjustments in linear RGB space
    adjusted_image = apply_adjustments(hald_image, adjustments)
    print(f"Adjusted image shape: {adjusted_image.shape}")
    
    if reference_lut_file:
        reference_lut = load_cube_file(reference_lut_file)
        if reference_lut.shape == (lut_size, lut_size, lut_size, 3):
            adjusted_image = apply_reference_adjustment(adjusted_image, reference_lut, adjustments)
            print(f"Adjusted image shape after reference adjustment: {adjusted_image.shape}")
            # Save the LUT directly from the blended LUT, not as an image
            image_to_cube(adjusted_image, lut_size, output_lut_file)
            print(f"Saved LUT to {output_lut_file}")
            return
        else:
            print(f"Reference LUT size {reference_lut.shape[0]} does not match target size {lut_size}, skipping reference adjustment")
    
    # Save adjusted Hald CLUT as image for verification (only if no reference LUT is used)
    Image.fromarray((adjusted_image * 255).astype(np.uint8)).save('adjusted_hald.png')
    print("Saved adjusted Hald CLUT as 'adjusted_hald.png' for verification")
    
    # Convert adjusted Hald CLUT to .cube LUT file
    image_to_cube(adjusted_image, lut_size, output_lut_file)
    print(f"Saved LUT to {output_lut_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        xmp_file_path = sys.argv[1]
        output_lut_path = sys.argv[2]
        reference_lut = sys.argv[3] if len(sys.argv) > 3 else None
    elif len(sys.argv) > 1:
        xmp_file_path = sys.argv[1]
        output_lut_path = "output_" + xmp_file_path.split("/")[-1].replace(".xmp", ".cube")
        reference_lut = None
    else:
        xmp_file_path = "example.xmp"
        output_lut_path = "output_xmp.cube"
        reference_lut = None
    
    print(f"Processing XMP file: {xmp_file_path}")
    xmp_to_lut(xmp_file_path, output_lut_path, reference_lut_file=reference_lut)
    print(f"LUT generated and saved as {output_lut_path}")