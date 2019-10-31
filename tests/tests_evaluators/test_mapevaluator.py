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
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from evaluators.map_evaluator import MAPEvaluator


class TestMapEvaluator(TestCase):

    def test_evaluate(self):
        # Arrange
        iou_evaluator = MagicMock()

        iou_evaluator.evaluate.side_effect = lambda t, p: torch.rand((t.shape[0], p.shape[0]))

        sut = MAPEvaluator(iou_evaluator)

        target = [{"image_id": 1,
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])
                   }]

        predicted = [{"image_id": 1,
                      "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                      "labels": torch.tensor([1]),
                      "scores": torch.tensor([1.0])
                      }]
        expected_map = 1

        # Act
        actual = sut.evaluate(target, predicted)

        # Assert
        self.assertEqual(expected_map, actual)
