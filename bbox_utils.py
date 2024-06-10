def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    min_distance = float('inf')
    closest_keypoint_index = None
    for keypoint_index in keypoint_indices:
        keypoint = (keypoints[keypoint_index*2], keypoints[keypoint_index*2+1])
        distance = calculate_distance(point, keypoint)
        if distance < min_distance:
            min_distance = distance
            closest_keypoint_index = keypoint_index
    return closest_keypoint_index

def get_height_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return y2 - y1

def measure_xy_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x2 - x1), abs(y2 - y1)

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))