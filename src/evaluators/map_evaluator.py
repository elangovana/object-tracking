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
import numpy as np
import torch
from sklearn.metrics import average_precision_score

from evaluators.base_detection_evaluator import BaseDetectionEvaluator


class MAPEvaluator(BaseDetectionEvaluator):
    """
    Calculates intersection over union pairwise between the ground truth and predictions matrix for the best match
    """

    def __init__(self, ioumatrix_evaluator, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.ioumatrix_evaluator = ioumatrix_evaluator

    def evaluate(self, g, p):
        """

        :param g: g is a list of dict. One dict represents the annotations of a singe image
        :param p: p is a list of dict
        :return: map score
        """

        assert len(g) == len(p), "The length of target {} and predicted {} mismatch ".format(g, p)

        target_class = []
        predicted_class = []
        predicted_score = []

        for i, (gi, pi) in enumerate(zip(g, p)):
            best_match_iou = torch.max(self.ioumatrix_evaluator.evaluate(gi["boxes"], pi["boxes"]), dim=1)
            best_match_iou_idx = best_match_iou[1]
            best_match_iou_score = best_match_iou[0]

            predicted_class_i = pi["labels"][best_match_iou_idx]

            confidence_score_i = pi["scores"][best_match_iou_idx]

            target_class_i = gi["labels"]

            index_below_iou_threshold = (best_match_iou_score < self.iou_threshold).nonzero()

            # Change class to background when below overlap to threshold so it is not counted
            predicted_class_i[index_below_iou_threshold] = 0

            predicted_class.extend(predicted_class_i)
            predicted_score.extend(confidence_score_i)
            target_class.extend(target_class_i)

        average_precision = self._get_average_precision(target_class, predicted_score)

        # Returns best Iou and the scores
        return average_precision

    def _get_average_precision(self, target_class, predicted_score):
        # For each class
        classes = np.unique(target_class)
        assert len(classes) <= 2, "This class only works with 2 classes"
        classes = 2
        #  precision = dict()
        #  recall = dict()
        #  average_precision = dict()
        #
        #  #target_class = label_binarize(target_class, classes=classes)
        # # for i in classes:
        #  precision, recall, _ = precision_recall_curve(target_class,
        #                                                      predicted_score)
        #  average_precision = average_precision_score(target_class, predicted_score)

        # A "micro-average": quantifying score on all classes jointly
        # precision["micro"], recall["micro"], _ = precision_recall_curve(target_class.ravel(),
        #                                                                 predicted_score.ravel())
        average_precision = average_precision_score(target_class, predicted_score,
                                                    average="micro")

        return average_precision
