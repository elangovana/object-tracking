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
from ddt import data, unpack, ddt

from evaluators.iou_matrix import IoUMatrix


@ddt
class TestIoUMatrix(TestCase):

    @data(([0, 1, 0, 1], [0, 1, 0, 1], 1.0)
        , ([0, 1, 0, 1], [1, 2, 1, 2], 0.0)
        , ([0, 2, 0, 2], [1, 2, 1, 2], .25)
        , ([1, 2, 1, 2], [0, 2, 0, 2], .25)
        , ([1, 2, 2, 3], [0, 2, 1, 3], .25)
          )
    @unpack
    def test_evaluate(self, g, p, expected_iou):
        # Arrange
        sut = IoUMatrix()
        g = torch.tensor([g]).float()
        p = torch.tensor([p]).float()

        # Act
        actual = sut.evaluate(g, p)

        # Assert
        self.assertEqual(expected_iou, round(actual.item(), 2))

    def test_evaluate_size_2(self):
        """
        Test case for matrix of size 2
        :return:
        """
        # Arrange
        p = [
            [1, 2, 1, 2],
            [2, 3, 2, 3]
        ]
        g = [
            [2, 3, 2, 3],
            [2, 3, 2, 3]
        ]

        expected_iou = torch.tensor([[0, 1],
                                     [0, 1]]).float()
        sut = IoUMatrix()
        g = torch.tensor(g).float()
        p = torch.tensor(p).float()

        # Act
        actual = sut.evaluate(g, p)

        # Assert
        self.assertTrue(torch.all(expected_iou.eq(actual)))

    def test_evaluate_uneven(self):
        """
        Test case for matrix of uneven sizes
        :return:
        """
        # Arrange
        p = [
            [1, 2, 1, 2]
        ]
        g = [
            [2, 3, 2, 3],
            [1, 2, 1, 2]
        ]

        expected_iou = torch.tensor([[0],
                                     [1]]).float()
        sut = IoUMatrix()
        g = torch.tensor(g).float()
        p = torch.tensor(p).float()

        # Act
        actual = sut.evaluate(g, p)

        print(actual)

        # Assert
        self.assertTrue(torch.all(expected_iou.eq(actual)))
