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
from evaluators.base_detection_evaluator import BaseDetectionEvaluator


class IoU(BaseDetectionEvaluator):
    """
    Calculates intersection over union
    """

    def __init__(self):
        pass

    def evaluate(self, gx1, gx2, gy1, gy2, px1, px2, py1, py2) -> float:
        assert gx1 < gx2
        assert gy1 < gy2
        assert px1 < px2
        assert py1 < py2

        start_x = max(gx1, px1)
        end_x = min(gx2, px2)

        start_y = max(gy1, py1)
        end_y = min(gy2, py2)

        interaction_area = (end_x - start_x) * (end_y - start_y)

        area_g = (gx2 - gx1) * (gy2 - gy1)
        area_p = (px2 - px1) * (py2 - py1)

        union_area = area_g + area_p - interaction_area

        iou = interaction_area / union_area

        return iou
