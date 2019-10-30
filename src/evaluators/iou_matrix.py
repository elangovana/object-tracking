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
import torch

from evaluators.base_detection_evaluator import BaseDetectionEvaluator


class IoUMatrix(BaseDetectionEvaluator):
    """
    Calculates intersection over union
    """

    def __init__(self):
        pass

    def evaluate(self, g, p) -> float:
        assert len(g.shape) == 2, "Expect a 2 d tensor for ground truth"
        assert len(p.shape) == 2, "Expect a 2 d tensor for predicted boxes"

        assert g.shape[1] == 4, "Expected 2d tensor of shape ( :, 4) but this tensor has shape {}".format(g.shape)
        assert p.shape[1] == 4, "Expected 2d tensor of shape ( :, 4) but this tensor has shape {}".format(p.shape)

        assert torch.all(g[:, 0] < g[:, 1]).item(), "Expect all items in index g[:,0] to be less than index g[:,1]"
        assert torch.all(g[:, 2] < g[:, 3]).item(), "Expect all items in index g[:,2] to be less than index g[:,3]"

        assert torch.all(p[:, 0] < p[:, 1]).item(), "Expect all items in index p[:,0] to be less than index p[:,1]"
        assert torch.all(p[:, 2] < p[:, 3]).item(), "Expect all items in index p[:,2] to be less than index p[:,3]"

        g = g.unsqueeze(1)

        start_x = torch.max(g[:, :, 0], p[:, 0])
        end_x = torch.min(g[:, :, 1], p[:, 1])

        start_y = torch.max(g[:, :, 2], p[:, 2])
        end_y = torch.min(g[:, :, 3], p[:, 3])

        interaction_area = (end_x - start_x) * (end_y - start_y)

        area_g = (g[:, :, 1] - g[:, :, 0]) * (g[:, :, 3] - g[:, :, 2])
        area_p = (p[:, 1] - p[:, 0]) * (p[:, 3] - p[:, 2])

        union_area = area_g + area_p - interaction_area

        iou = interaction_area / union_area

        return iou
