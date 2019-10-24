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

from datasets.mots17_dataset import Mots17Dataset


class TestMots17Dataset(TestCase):
    def test_forward(self):
        # Arrange
        input_path = os.path.join(os.path.dirname(__file__), "..", "data", "clips")
        sut = Mots17Dataset(root=input_path, annotation_path=input_path, frames_per_clip=30)
        expected_num_frames = 900
        expected_num_identities = 91

        # Act
        video, label = sut[0]

        # Assert
        self.assertEqual(expected_num_frames, len(video))
        self.assertEqual(expected_num_identities, len(label))
