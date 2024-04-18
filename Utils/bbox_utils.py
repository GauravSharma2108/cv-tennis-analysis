def get_center_of_bbox(bbox):
    """Get the center of the bounding box

    Args:
        bbox (list): list containing the coordinates of the bounding box in the format [x1, y1, x2, y2]

    Returns:
        tuple: tuple containing the coordinates of the center of the bounding box in the format (x_center, y_center)
    """
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return (x_center, y_center)

def measure_distance(p1, p2):
    """Get the Euclidean distance between two points

    Args:
        p1 (tuple): tuple containing the coordinates of the first point in the format (x1, y1)
        p2 (tuple): tuple containing the coordinates of the second point in the format (x2, y2)

    Returns:
        float: Euclidean distance between the two points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5