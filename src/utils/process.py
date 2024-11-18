from typing import List, Union
import cupy as cp
import numpy as np

def preprocess(
    image_data: cp.ndarray, 
    dtype: Union[np.dtype, type] = cp.float32,
    normalize_factor: float = 255.0, 
    channel_first: bool = True,
    batched: bool = True) -> cp.ndarray:
    
    image_normalized: cp.ndarray = cp.asarray(image_data).astype(dtype) / normalize_factor
    
    if channel_first:
        p_image: cp.ndarray = cp.transpose(image_normalized, (2, 0, 1))
    else:
        p_image: cp.ndarray = image_normalized
    
    if batched:
        result_image: cp.ndarray = cp.expand_dims(p_image, axis=0)
    else:
        result_image: cp.ndarray = p_image
    
    return result_image

def postprocess(
    output_data: Union[cp.ndarray],
    scale_width: float,
    scale_height: float,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5) -> List[List[float]]:
    nms_results: List[List[float]] = nms(output_data, score_threshold=score_threshold, iou_threshold=iou_threshold)

    scaled_bboxes: List[List[float]] = []
    for bbox in nms_results:
        class_id, score, x_min, y_min, x_max, y_max = bbox
        new_x_min: float = x_min * scale_width
        new_y_min: float = y_min * scale_height
        new_x_max: float = x_max * scale_width
        new_y_max: float = y_max * scale_height
        scaled_bbox: List[float] = [class_id, score, new_x_min, new_y_min, new_x_max, new_y_max]
        scaled_bboxes.append(scaled_bbox)
    
    return scaled_bboxes


def nms(
    detections: cp.ndarray,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5) -> List[List[float]]:
    
    scores: cp.ndarray = detections[:, 1]
    keep: cp.ndarray = scores >= score_threshold
    detections: cp.ndarray = detections[keep]
    scores: cp.ndarray = detections[:, 1]
    boxes: cp.ndarray = detections[:, 2:]

    idxs: cp.ndarray = cp.argsort(scores)[::-1]
    selected_idxs: List[int] = []

    while len(idxs) > 0:
        current_idx: int = idxs[0]
        selected_idxs.append(current_idx)

        if len(idxs) == 1:
            break

        current_box: cp.ndarray = boxes[current_idx].reshape(1, -1)
        rest_boxes: cp.ndarray = boxes[idxs[1:]]
        ious: cp.ndarray = iou(current_box, rest_boxes).squeeze()

        idxs = idxs[1:][ious < iou_threshold]
        selected_detections = detections[cp.array(selected_idxs)].get()
    
    return selected_detections.tolist()

def iou(boxA: cp.ndarray, boxB: cp.ndarray) -> cp.ndarray:
    xA = cp.maximum(boxA[:, None, 0], boxB[None, :, 0])
    yA = cp.maximum(boxA[:, None, 1], boxB[None, :, 1])
    xB = cp.minimum(boxA[:, None, 2], boxB[None, :, 2])
    yB = cp.minimum(boxA[:, None, 3], boxB[None, :, 3])

    interArea = cp.maximum(xB - xA, 0) * cp.maximum(yB - yA, 0)
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    return interArea / (boxAArea[:, None] + boxBArea[None, :] - interArea)

#     with cp.cuda.Device(gpu).use():
#         # Example usage
#         output_data = cp.array([[0, 0.9, 10, 10, 50, 50], [0, 0.8, 15, 15, 55, 55]], dtype=cp.float32)
#         scale_width, scale_height = 1.0, 1.0
#         results = postprocess(output_data, scale_width, scale_height)
#         print(results)

# if __name__ == "__main__":
#     main(gpu=0)  # Replace with the appropriate GPU ID
