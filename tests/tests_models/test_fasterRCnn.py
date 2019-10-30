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

from models.faster_rcnn import FasterRCnn


class TestFasterRCnn(TestCase):
    def test_forward(self):
        """
        Simple test to make sure that the code executes
        :return:
        """
        # Arrange
        num_classes = 2
        sut = FasterRCnn(num_classes)
        input = torch.rand((1, 3, 224, 224))
        target = [{"image_id": 1,
                   "boxes": torch.tensor([[1, 2, 3, 4]]).float(),
                   "labels": torch.tensor([1]),
                   "area": torch.tensor([1.0]),
                   "iscrowd": torch.tensor([0])}]

        # Act
        actual = sut(input, target)

        # Assert
        self.assertIsNotNone(actual)
