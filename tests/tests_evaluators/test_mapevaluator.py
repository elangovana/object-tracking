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

import torch

from evaluators.map_evaluator import MAPEvaluator


class TestMapEvaluator(TestCase):

    def test_evaluate_single_full_match(self):
        """
        Test simple case
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),

                   "iscrowd": torch.tensor([0])
                   }]

        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                      "labels": torch.tensor([1]),
                      "scores": torch.tensor([1.0])
                      }]
        expected_map_score = 1.0

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(expected_map_score, round(actual, 2))

    def test_evaluate_2_gt_boxes_incorrect_confidence(self):
        """
        Test case where 2 gt, 1 boxes were predicted with one matches, and has incorrect confidence score
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14]]).float(),
                   "labels": torch.tensor([1, 1]),
                   "iscrowd": torch.tensor([0, 0])
                   }]

        # [1, 2, 3, 4]
        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                      "labels": torch.tensor([1]),
                      "scores": torch.tensor([0.8])
                      }]
        expected_map = .5

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(round(expected_map, 2), round(actual, 2))

    def test_evaluate_2_gt_2_p_boxes_one_incorrect(self):
        """
        Test case where 2 gt, 2 boxes were predicted and only one matches, and has incorrect confidence score
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14]]).float(),
                   "labels": torch.tensor([1, 1]),
                   "iscrowd": torch.tensor([0, 0])
                   }]

        # [1, 2, 3, 4]
        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[19, 20, 22, 23], [1, 2, 3, 4]]).float(),
                      "labels": torch.tensor([1, 1]),
                      "scores": torch.tensor([1, 1])
                      }]
        expected_map = .25

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(round(expected_map, 2), round(actual, 2))

    def test_evaluate_2_gt_2_p_boxes_both_correct_confidence(self):
        """
        Test case where 2 gt, 2 boxes were predicted and only one matches, and has incorrect confidence score
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14]]).float(),
                   "labels": torch.tensor([1, 1]),
                   "iscrowd": torch.tensor([0, 0])
                   }]

        # [1, 2, 3, 4]
        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[11, 12, 13, 14], [1, 2, 3, 4]]).float(),
                      "labels": torch.tensor([1, 1]),
                      "scores": torch.tensor([1, 1])
                      }]
        expected_map = 1.0

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(round(expected_map, 2), round(actual, 2))

    def test_evaluate_2_p_boxes_incorrect_confidence(self):
        """
        Test case where 2 boxes were predicted and only one matches, and has incorrect confidence score
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])
                   }]

        # [1, 2, 3, 4]
        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4], ]).float(),
                      "labels": torch.tensor([1, 1]),
                      "scores": torch.tensor([0.8, 0.7])
                      }]
        expected_map = .5

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(round(expected_map, 2), round(actual, 2))

    def test_evaluate_2_p_boxes_correct_confidence(self):
        """
        Test case where 2 boxes were predicted and only one matches, and has correct confidence score
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])
                   }]

        # [1, 2, 3, 4]
        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4], ]).float(),
                      "labels": torch.tensor([1, 1]),
                      "scores": torch.tensor([0.5, 0.7])
                      }]
        expected_map = 1.0

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(round(expected_map, 2), round(actual, 2))

    def test_evaluate_2_images(self):
        """
        Test case where there are 2 images as input
        :return:
        """
        # Arrange

        sut = MAPEvaluator()

        target = [{"image_id": torch.tensor(1),
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])
                   },
                  {"image_id": torch.tensor(2),
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])
                   }]

        predicted = [{"image_id": torch.tensor(1),
                      "boxes": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).float(),
                      "labels": torch.tensor([1, 1]),
                      "scores": torch.tensor([1.0, 1.0])
                      },
                     {"image_id": torch.tensor(2),
                      "boxes": torch.tensor([[7, 8, 9, 10]]).float(),
                      "labels": torch.tensor([1]),
                      "scores": torch.tensor([1.0])
                      }]
        expected_map = .5

        # Act
        actual = sut(target, predicted)

        # Assert
        self.assertEqual(expected_map, round(actual, 2))
