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

    def __init__(self, iou_threshold=0.5):
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

        cocoEval.params.imgIds = coco_gt.getImgIds()
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        # Returns best Iou and the scores
        return cocoEval.stats[0]
