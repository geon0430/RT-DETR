# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import itertools
import json
from ppdet.metrics.json_results import get_det_res, get_det_poly_res, get_seg_res, get_solov2_segm_res, get_keypoint_res, get_pose3d_res
from ppdet.metrics.map_utils import draw_pr_curve

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)


def get_infer_results(outs, catid, bias=0, save_threshold=0):
    """
    Get result at the stage of inference.
    The output format is dictionary containing bbox or mask result.

    For example, bbox result is a list and each element contains
    image_id, category_id, bbox and score.
    """
    if outs is None or len(outs) == 0:
        raise ValueError(
            'The number of valid detection result if zero. Please use reasonable model and check input data.'
        )

    im_id = outs['im_id']
    im_file = outs['im_file'] if 'im_file' in outs else None

    infer_res = {}
    if 'bbox' in outs:
        if len(outs['bbox']) > 0 and len(outs['bbox'][0]) > 6:
            infer_res['bbox'] = get_det_poly_res(
                outs['bbox'], outs['bbox_num'], im_id, catid, bias=bias)
        else:
            infer_res['bbox'] = get_det_res(
                outs['bbox'],
                outs['bbox_num'],
                im_id,
                catid,
                bias=bias,
                im_file=im_file,
                save_threshold=save_threshold)

    if 'mask' in outs:
        # mask post process
        infer_res['mask'] = get_seg_res(outs['mask'], outs['bbox'],
                                        outs['bbox_num'], im_id, catid)

    if 'segm' in outs:
        infer_res['segm'] = get_solov2_segm_res(outs, im_id, catid)

    if 'keypoint' in outs:
        infer_res['keypoint'] = get_keypoint_res(outs, im_id)
        outs['bbox_num'] = [len(infer_res['keypoint'])]

    if 'pose3d' in outs:
        infer_res['pose3d'] = get_pose3d_res(outs, im_id)
        outs['bbox_num'] = [len(infer_res['pose3d'])]

    return infer_res

def cocoapi_eval(jsonfile,
                 style,
                 coco_gt=None,
                 anno_file=None,
                 max_dets=(100, 300, 1000),
                 classwise=False,
                 sigmas=None,
                 use_area=True):
    """
    Args:
        jsonfile (str): 모델의 예측 결과를 COCO 형식으로 저장한 JSON 파일 경로
        style (str): bbox, segm, 등 구분 (데모에서는 bbox만 처리한다고 가정)
        coco_gt (None): 기존 PaddleDetection에서 넘기는 파라미터 (사용 X)
        anno_file (str): Ground Truth COCO JSON 파일 경로
        max_dets (tuple): 최대 Dets 개수 (데모에서는 사용하지 않음)
        classwise (bool): 클래스별 AP를 출력할지 여부
        sigmas (nparray): keypoint용 파라미터 (데모에서는 사용 X)
        use_area (bool): area 필드 사용 여부 (데모에서는 사용 X)

    Returns:
        list: 간단히 [mAP] 형태로 반환 (COCOeval.stats와 유사한 구조)
    """

    if anno_file is None:
        raise ValueError("anno_file is required for custom evaluation.")
    with open(anno_file, 'r') as f:
        gt_data = json.load(f)
    gt_annotations = gt_data["annotations"]      
    gt_categories = gt_data["categories"]         
    cat_ids = [cat["id"] for cat in gt_categories]

    with open(jsonfile, 'r') as f:
        pred_data = json.load(f)
    gt_dict = {}
    for ann in gt_annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        bbox   = ann["bbox"]  # [x, y, w, h]
        if img_id not in gt_dict:
            gt_dict[img_id] = {}
        if cat_id not in gt_dict[img_id]:
            gt_dict[img_id][cat_id] = []
        gt_dict[img_id][cat_id].append(bbox)

    pred_dict = {}
    for pd in pred_data:
        img_id = pd["image_id"]
        cat_id = pd["category_id"]
        bbox   = pd["bbox"]  
        score  = pd["score"]
        if img_id not in pred_dict:
            pred_dict[img_id] = {}
        if cat_id not in pred_dict[img_id]:
            pred_dict[img_id][cat_id] = []
        pred_dict[img_id][cat_id].append((bbox, score))

    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    category_aps = {} 
    for cat_id in cat_ids:
        category_aps[cat_id] = []

    for iou_thr in iou_thresholds:
        ap_per_cat = {}  

        for cat_id in cat_ids:
            all_preds = [] 
            total_gt = 0  

            for img_id, cats_dict in gt_dict.items():
                if cat_id in cats_dict:
                    total_gt += len(cats_dict[cat_id])

            for img_id, cats_dict in pred_dict.items():
                if cat_id in cats_dict:
                    for (bbox, score) in cats_dict[cat_id]:
                        all_preds.append((img_id, bbox, score))

            all_preds.sort(key=lambda x: x[2], reverse=True)
            matched_gts = {}  

            tp_list = []
            fp_list = []

            for (img_id, pbbox, pscore) in all_preds:
                if img_id not in gt_dict or cat_id not in gt_dict[img_id]:
                    tp_list.append(0)
                    fp_list.append(1)
                    continue

                gt_bboxes = gt_dict[img_id][cat_id]
                iou_max = 0.0
                match_gt_idx = -1

                for gt_idx, gtbbox in enumerate(gt_bboxes):
                    if (img_id, gt_idx) in matched_gts:
                        continue 
                    iou = compute_iou(pbbox, gtbbox)
                    if iou > iou_max:
                        iou_max = iou
                        match_gt_idx = gt_idx

                if iou_max >= iou_thr and match_gt_idx != -1:
                    tp_list.append(1)
                    fp_list.append(0)
                    matched_gts[(img_id, match_gt_idx)] = True
                else:
                    tp_list.append(0)
                    fp_list.append(1)

            tp_array = np.cumsum(tp_list)
            fp_array = np.cumsum(fp_list)
            if len(tp_array) == 0:
                ap_per_cat[cat_id] = 0.0
                continue

            recalls = tp_array / float(total_gt) if total_gt else np.zeros_like(tp_array)
            precisions = tp_array / (tp_array + fp_array + 1e-16)

            ap_value = 0.0
            prev_recall = 0.0
            for i in range(len(precisions)):
                recall = recalls[i]
                prec = precisions[i]
                ap_value += prec * (recall - prev_recall)
                prev_recall = recall
            ap_per_cat[cat_id] = ap_value

        for cat_id in cat_ids:
            category_aps[cat_id].append(ap_per_cat[cat_id])

    final_maps = []
    for cat_id in cat_ids:
        final_maps.append(np.mean(category_aps[cat_id]))
    overall_map = np.mean(final_maps)

    logger.info("========================================")
    logger.info(f" mAP  = {overall_map:.4f}")
    logger.info("========================================")

    if classwise:
        from terminaltables import AsciiTable
        results_per_category = []
        for cat_id in cat_ids:
            cat_info = next((c for c in gt_categories if c["id"] == cat_id), None)
            cat_name = cat_info["name"] if cat_info else str(cat_id)
            cat_ap = np.mean(category_aps[cat_id])
            results_per_category.append((cat_name, f"{cat_ap:.3f}"))

        num_columns = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        headers = ["Category", "AP"] * (num_columns // 2)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)]
        )
        table_data = [headers]
        table_data += [result for result in results_2d]
        from terminaltables import AsciiTable
        table = AsciiTable(table_data)
        logger.info("Per-category AP:\n{}".format(table.table))

    sys.stdout.flush()

    return [overall_map] + [0.0] * 11


def compute_iou(bbox1, bbox2):
    """
    COCO 형식의 bbox: [x, y, w, h]
    IoU를 계산하기 위한 헬퍼 함수
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def json_eval_results(metric, json_directory, dataset):
    """
    cocoapi eval with already exists proposal.json, bbox.json or mask.json
    """
    assert metric == 'COCO'
    anno_file = dataset.get_anno()
    json_file_list = ['proposal.json', 'bbox.json', 'mask.json']
    if json_directory:
        assert os.path.exists(
            json_directory), "The json directory:{} does not exist".format(
                json_directory)
        for k, v in enumerate(json_file_list):
            json_file_list[k] = os.path.join(str(json_directory), v)

    coco_eval_style = ['proposal', 'bbox', 'segm']
    for i, v_json in enumerate(json_file_list):
        if os.path.exists(v_json):
            cocoapi_eval(v_json, coco_eval_style[i], anno_file=anno_file)
        else:
            logger.info("{} not exists!".format(v_json))
