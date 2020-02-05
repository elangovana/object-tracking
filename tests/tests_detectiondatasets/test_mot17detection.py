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

from detection_datasets.mot17_detection import Mot17Detection


class TestMot17Detection(TestCase):
    def test_getitem(self):
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "small_clips")
        sut = Mot17Detection(root=input_path)
        expected_num_boxes = 2
        index = 0

        # Act
        video, label = sut[index]

        # Assert
        self.assertEqual(expected_num_boxes, len(label["boxes"]))
        self.assertEqual(expected_num_boxes, len(label["iscrowd"]))
        self.assertEqual(expected_num_boxes, len(label["area"]))
        self.assertEqual(index + 1, label["image_id"])
        self.assertEqual(expected_num_boxes, len(label["labels"]))

        self.assertIsInstance(video, str)

    def test_getitem_lengthsmatch(self):
        """
        Testcase: Ensures that the length of each value in the returned label matches
        :return:
        """
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
        sut = Mot17Detection(root=input_path)

        index = 0

        # Act
        video, label = sut[index]

        # Assert
        self.assertEqual(len(label["iscrowd"]), len(label["boxes"]))
        self.assertEqual(len(label["iscrowd"]), len(label["iscrowd"]))
        self.assertEqual(len(label["iscrowd"]), len(label["area"]))
        self.assertEqual(index + 1, label["image_id"])
        self.assertEqual(len(label["iscrowd"]), len(label["labels"]))

        self.assertIsInstance(video, str)

    def test_len(self):
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
        sut = Mot17Detection(root=input_path)
        expected_len = 11

        # Act
        actual_len = len(sut)

        # Assert
        self.assertEqual(actual_len, expected_len)
