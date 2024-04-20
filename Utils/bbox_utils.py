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

def get_foot_position(bbox):
    """Get the foot position of the bounding box

    Args:
        bbox (list): list containing the coordinates of the bounding box in the format [x1, y1, x2, y2]

    Returns:
        tuple: tuple containing the coordinates of the foot position of the bounding box in the format (x_center, y2)
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    """Get the index of the closest keypoint to a given point, from a list of keypoints provided

    Args:
        point: foot of the player bbox
        keypoints: list of keypoints
        keypoint_indices: list of keypoint indices to consider

    Returns:
        int: index of the closest keypoint
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_index in keypoint_indices:
        keypoint = keypoints[keypoint_index*2], keypoints[keypoint_index*2+1]
        distance = abs(point[1]-keypoint[1])

        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index
    
    return key_point_ind

def get_height_of_bbox(bbox):
    """Get the height of the bounding box

    Args:
        bbox (list): list containing the coordinates of the bounding box in the format [x1, y1, x2, y2]

    Returns:
        float: height of the bounding box
    """
    return bbox[3]-bbox[1]

def measure_xy_distance(p1,p2):
    """Get the Euclidean distance between two points

    Args:
        p1 (tuple): tuple containing the coordinates of the first point in the format (x1, y1)
        p2 (tuple): tuple containing the coordinates of the second point in the format (x2, y2)

    Returns:
        tuple: tuple containing the Euclidean distance between the two points in the format (x_distance, y_distance)
    """
    return abs(p1[0]-p2[0]), abs(p1[1]-p2[1])

def get_center_of_bbox(bbox):
    """Get the center of the bounding box

    Args:
        bbox (list): list containing the coordinates of the bounding box in the format [x1, y1, x2, y2]

    Returns:
        tuple: tuple containing the coordinates of the center of the bounding box in the format (x_center, y_center)
    """
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))