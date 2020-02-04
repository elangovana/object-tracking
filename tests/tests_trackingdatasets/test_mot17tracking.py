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

import torch

from tracking_datasets.mot17_tracking import Mot17Tracking


class TestMot17Tracking(TestCase):
    def test_getitem(self):
        """
        Ensure that get item returns the correct number of boxes, clips and labels
        :return:
        """
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "small_clips")
        sut = Mot17Tracking(root=input_path)
        expected_num_boxes = 1
        expected_clip_len = 1
        index = 0

        # Act
        clips, boxes, label = sut[index]

        # Assert
        self.assertEqual(expected_num_boxes, len(boxes), "Expected number of boxes incorrect")
        self.assertEqual(len(boxes), len(label), "Expected the number of boxes and labels to match")
        self.assertEqual(expected_clip_len, len(clips), "Expected the number frames in clip mismatch")
        self.assertEqual(len(boxes), len(clips), "Expected the number of boxes and clip length to match")

    def test_getitem_ensure_clip_is_in_sequence(self):
        """
        Ensure that the clips are returned in order
        :return:
        """
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "one_clip")
        sut = Mot17Tracking(root=input_path)
        expected_clips_in_order = [os.path.join(input_path, "MOT17-11-SDP", "img1", f) for f in ["000001.jpg",
                                                                                                 "000002.jpg",
                                                                                                 "000003.jpg",
                                                                                                 "000004.jpg",
                                                                                                 "000005.jpg"]]
        index = 0

        # Act
        actual_clips, _, _ = sut[index]

        # Assert
        self.assertSequenceEqual(expected_clips_in_order, actual_clips)

    def test_getitem_ensure_boundingboxes_per_clip_frame_are_correct(self):
        """
        Ensure that the clips and the corresponding bounding boxes are returned correctly
        :return:
        """
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "one_clip")
        sut = Mot17Tracking(root=input_path)
        expected_bounding_boxes = torch.tensor([[867, 145, 236 + 867, 635 + 145], [-33, 10, 385 - 33, 1122 + 10]])
        frame_index = 0
        clip_index = 0

        # Act
        _, actual_bounding_boxes, _ = sut[clip_index]

        # Assert
        self.assertTrue(torch.all(expected_bounding_boxes.eq(actual_bounding_boxes[frame_index])),
                        " The bounding boxes {} do not match the expected {}".format(actual_bounding_boxes[frame_index],
                                                                                     expected_bounding_boxes))

    def test_getitem_ensure_ids_per_clip_frame_are_correct(self):
        """
        Ensure that the clips and the corresponding bounding boxes are returned correctly
        :return:
        """
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "one_clip")
        sut = Mot17Tracking(root=input_path)
        expected_ids = torch.tensor([1, 2])
        frame_index = 0
        clip_index = 0

        # Act
        _, _, labels = sut[clip_index]

        # Assert
        self.assertTrue(torch.all(expected_ids.eq(labels[frame_index])),
                        " The bounding boxes {} do not match the expected {}".format(labels[frame_index],
                                                                                     expected_ids))

    def test_len(self):
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
        sut = Mot17Tracking(root=input_path)
        expected_len = 2

        # Act
        actual_len = len(sut)

        # Assert
        self.assertEqual(actual_len, expected_len)
