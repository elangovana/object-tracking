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

import torch

from evaluators.base_detection_evaluator import BaseDetectionEvaluator


class MAPEvaluator(BaseDetectionEvaluator):
    """
    Calculates intersection over union pairwise between the ground truth and predictions matrix for the best match.
    TODO: This is sloppy need to clean and correct the calc a.. doesnt implement threshold based mean average precsion

    Resources:
        https://github.com/rafaelpadilla/Object-Detection-Metrics
    """

    def __init__(self, ioumatrix_evaluator, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.ioumatrix_evaluator = ioumatrix_evaluator

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

        # get unique ground truth classes
        classes = []
        for gi in g:
            classes.append(gi["labels"])

        classes = set(classes)

        # TODO: extend to more than one class
        tp, fp, fn = self.get_confusion_matix(g, p, 1)

        average_precision = self._get_average_precision(tp, fp, fn)

        # Returns best Iou and the scores
        return average_precision

    def get_confusion_matix(self, g, p, class_index):
        target_class = []
        predicted_class = []
        predicted_score = []

        # True Positive (TP): A correct detection. Detection with IOU ≥ threshold
        # False Positive (FP): A wrong detection. Detection with IOU < threshold
        # False Negative (FN): A ground truth not detected

        tp = []
        fp = []
        fn = []
        gt_classes = []
        for i, (gi, pi) in enumerate(zip(g, p)):
            iou_g_vs_p = self.ioumatrix_evaluator(gi["boxes"], pi["boxes"])

            self.logger.debug("Number of boxes predicted {}".format(len(pi["boxes"])))

            # # Find the max area for each gt object
            # best_match_iou = torch.max(iou_g_vs_p, dim=1)[0]
            #
            # # Obtain indexes of the best match for gt and pt
            # mask_best_match_indices = (iou_g_vs_p == best_match_iou.unsqueeze(dim=1)).nonzero()
            #
            # # Predictions greater than & less than threshold
            # # Potential TP
            # mask_gt_threshold = (best_match_iou >= self.iou_threshold)
            #
            # # Get index of items gt threshold
            # mask_gt_threshold_index = (best_match_iou >= self.iou_threshold).nonzero()
            #
            # #  FP
            # mask_lt_threshold = ~ mask_gt_threshold
            #
            # mask_lt_threshold_index = ~ mask_gt_threshold.nonzero()
            #
            # # Only get classes for greater than Iou, everything else is a FP any
            # class_g_threshold = torch.masked_select(gi["labels"], mask_gt_threshold_index[0, :])

            # class_p_threshold = torch.masked_select(p["labels"], mask_gt_threshold_index[1, :])
            used_predicted_indices = set()
            used_gt_indices = set()
            for i in range(iou_g_vs_p.shape[0]):
                gt_row = iou_g_vs_p[i]
                best_match_iou = torch.max(gt_row, dim=0)[0]
                best_match_p_indx = torch.max(gt_row, dim=0)[1]

                best_match_p_class = pi["labels"][best_match_p_indx]
                best_match_g_class = gi["labels"][i]

                gt_classes.append([best_match_g_class])

                confidence = pi["scores"][best_match_p_indx]

                # False negative when index has already been matched against a different gt, so matching object for this gt
                if best_match_p_indx.item() in used_predicted_indices:
                    fn.append([confidence, best_match_g_class])
                    continue

                if best_match_iou >= self.iou_threshold and best_match_p_class == best_match_g_class:
                    tp.append([confidence, best_match_p_class])
                else:
                    fp.append([confidence, best_match_p_class])

                used_predicted_indices = used_predicted_indices.union([best_match_p_indx.item()])
                used_gt_indices = used_gt_indices.union([i])

            # For extra images in pred
            predicted_count = iou_g_vs_p.shape[1]
            for idx in list(set(range(predicted_count)) - used_predicted_indices):
                p_class = pi["labels"][idx]

                confidence = pi["scores"][idx]

                fp.append([confidence, p_class])

            # For images with no predictions
            gt_count = iou_g_vs_p.shape[0]
            for idx in list(set(range(gt_count)) - used_gt_indices):
                g_class = gi["labels"][idx]
                fp.append([0, g_class])

        return tp, fp, fn

    def _get_average_precision(self, tp, fp, fn):
        classes_score = {}
        self.reshape_by_class(classes_score, tp, "TP")
        self.reshape_by_class(classes_score, fp, "FP")
        self.reshape_by_class(classes_score, fn, "FN")

        f1 = 0
        for c, v in classes_score.items():
            precision = len(v["TP"]) / (len(v["TP"]) + len(v["FP"]))

            recall = len(v["TP"]) / (len(v["TP"]) + len(v["FN"]))

            if (precision + recall) > 0:
                f1 += 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 0

        average = f1 / len(classes_score)
        return average

    def reshape_by_class(self, classes_score, tp, detection_type):
        assert detection_type in ["TP", "FP", "FN"]

        for item in tp:
            label = item[1].item()
            confidence = item[0].item()

            if label not in classes_score:
                classes_score[label] = {"TP": [], "FP": [], "FN": []}

            classes_score[label][detection_type].append([confidence, label])
