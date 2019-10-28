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
import os
from unittest import TestCase

from datasets.mot17_detection import Mot17Detection


class TestMot17Detection(TestCase):
    def test_forward(self):
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
        sut = Mot17Detection(root=input_path)
        expected_num_boxes = 2

        # Act
        video, label = sut[0]

        # Assert
        self.assertEqual(expected_num_boxes, len(label["px"]))
        self.assertEqual(expected_num_boxes, len(label["py"]))
        self.assertEqual(expected_num_boxes, len(label["pw"]))
        self.assertEqual(expected_num_boxes, len(label["ph"]))

    def test_len(self):
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
        sut = Mot17Detection(root=input_path)
        expected_len = 11

        # Act
        actual_len = len(sut)

        # Assert
        self.assertEqual(actual_len, expected_len)
