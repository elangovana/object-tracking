# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************
import logging

from pycocotools.cocoeval import COCOeval

from evaluators.base_detection_evaluator import BaseDetectionEvaluator
from evaluators.coco_object_detection_api_adapter import CocoObjectDetectionApiAdapter

ANN_TYPE = "bbox"


class MAPEvaluator(BaseDetectionEvaluator):
    """
    Wrapper over coco eval
    """

    def __init__(self, iou_threshold=0.5, max_detections_per_image=500):
        self.max_detections_per_image = max_detections_per_image
        self.iou_threshold = iou_threshold

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, g, p):
        """

        :param g: g is a list of dict. One dict represents the annotations of a singe image
        :param p: p is a list of dict
        :return: map score
        """

        assert len(g) == len(p), "The length of target {} and predicted {} mismatch ".format(g, p)

        coco_gt, coco_p = CocoObjectDetectionApiAdapter().to_coco(g, p)

        cocoEval = COCOeval(coco_gt, coco_p, ANN_TYPE)
        cocoEval.params.maxDets = [1, self.max_detections_per_image // 2, self.max_detections_per_image]

        cocoEval.params.imgIds = coco_gt.getImgIds()
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # Returns best Iou and the scores
        # internally coco stats
        # stats = np.zeros((12,))
        # stats[0] = _summarize(1)
        # stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

        # returning 1, which is precision 1oU at 0.5
        return cocoEval.stats[1]
