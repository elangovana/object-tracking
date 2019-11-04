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

    def test_evaluate_single_full_match(self):
        """
        Test simple case
        :return:
        """
        # Arrange
        iou_evaluator = MagicMock()

        iou_evaluator.evaluate.side_effect = lambda t, p: torch.tensor([[1.0]])

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

    def test_evaluate_2_boxes(self):
        """
        Test case where 2 boxes were predicted and only one matches
        :return:
        """
        # Arrange
        iou_evaluator = MagicMock()

        iou_evaluator.evaluate.side_effect = lambda t, p: torch.tensor([[1.0, 0]])

        sut = MAPEvaluator(iou_evaluator)

        target = [{"image_id": 1,
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])
                   }]

        predicted = [{"image_id": 1,
                      "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).float(),
                      "labels": torch.tensor([1, 1]),
                      "scores": torch.tensor([1.0, 1.0])
                      }]
        expected_map = .667

        # Act
        actual = sut.evaluate(target, predicted)

        # Assert
        self.assertEqual(round(expected_map, 2), round(actual, 2))

    # def test_evaluate_2_images(self):
    #     """
    #     Test case where there are 2 images as input
    #     :return:
    #     """
    #     # Arrange
    #     iou_evaluator = MagicMock()
    #
    #     iou = {1: torch.tensor([[1.0, 0]])
    #         , 2:torch.tensor([[ 0]])}
    #
    #     counter = 0
    #     def mock_iou_func(t, p):
    #         result = iou[counter]
    #         return iou[counter]
    #
    #     iou_evaluator.evaluate.side_effect =  mock_iou_func
    #
    #     sut = MAPEvaluator(iou_evaluator)
    #
    #     target = [{"image_id": 1,
    #                "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
    #                "labels": torch.tensor([1]),
    #                "area": torch.tensor([1.0]),
    #                "iscrowd": torch.tensor([0])
    #                },
    #               {"image_id": 2,
    #                "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
    #                "labels": torch.tensor([1]),
    #                "area": torch.tensor([1.0]),
    #                "iscrowd": torch.tensor([0])
    #                }]
    #
    #     predicted = [{"image_id": 1,
    #                   "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).float(),
    #                   "labels": torch.tensor([1, 1]),
    #                   "scores": torch.tensor([1.0, 1.0])
    #                   },
    #                  {"image_id": 2,
    #                   "boxes": torch.tensor([[7, 8, 9, 10]]).float(),
    #                   "labels": torch.tensor([1]),
    #                   "scores": torch.tensor([1.0])
    #                   }]
    #     expected_map = .5
    #
    #     # Act
    #     actual = sut.evaluate(target, predicted)
    #
    #     # Assert
    #     self.assertEqual(expected_map, actual)
