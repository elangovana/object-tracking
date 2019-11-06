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

from ddt import data, unpack, ddt

from evaluators.iou import IoU


@ddt
class TestIoU(TestCase):

    @data((0, 1, 0, 1, 0, 1, 0, 1, 1.0)
        , (0, 1, 0, 1, 1, 2, 1, 2, 0.0)
        , (0, 2, 0, 2, 1, 2, 1, 2, .25)
        , (1, 2, 1, 2, 0, 2, 0, 2, .25)
        , (1, 2, 2, 3, 0, 2, 1, 3, .25)
          )
    @unpack
    def test_evaluate(self, gx1, gx2, gy1, gy2, px1, px2, py1, py2, expected_iou):
        # Arrange
        sut = IoU()

        # Act
        actual = sut.evaluate(gx1, gy1, gx2, gy2, px1, py1, px2, py2)

        # Assert
        self.assertEqual(expected_iou, round(actual, 2))
