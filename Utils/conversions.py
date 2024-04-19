def convert_pixel_distance_to_meters(pixel_distance, reference_height_in_meters, reference_height_in_pixels):
    """Converts a distance in pixels to meters

    Args:
        pixel_distance: distance in pixels
        reference_height_in_meters: reference height in meters
        reference_height_in_pixels: reference height in pixels

    Returns:
        distance in meters
    """
    return (pixel_distance * reference_height_in_meters) / reference_height_in_pixels

def convert_meters_to_pixel_distance(meters, reference_height_in_meters, reference_height_in_pixels):
    """Converts a distance in meters to pixels

    Args:
        meters: distance in meters
        reference_height_in_meters: reference height in meters
        reference_height_in_pixels: reference height in pixels

    Returns:
        distance in pixels
    """
    return (meters * reference_height_in_pixels) / reference_height_in_meters