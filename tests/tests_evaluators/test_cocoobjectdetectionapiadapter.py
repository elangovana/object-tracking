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

from evaluators.coco_object_detection_api_adapter import CocoObjectDetectionApiAdapter


class TestCocoObjectDetectionApiAdapter(TestCase):

    def test_to_coco_single_item(self):
        """
        Test case to check a single image with single box is mapped to coco object correctly
        :return:
        """
        # Arrange
        sut = CocoObjectDetectionApiAdapter()

        image_id = 99
        category_id = 1
        bbox = [1, 2, 10, 10]
        image1_anno = {"image_id": torch.tensor(image_id),
                       "boxes": torch.tensor([bbox]),
                       "labels": torch.tensor([category_id]),
                       "iscrowd": torch.tensor([0])}
        input = [image1_anno]

        expected_coco_annotation = [{'area': 100,
                                     'bbox': bbox,
                                     'category_id': category_id,
                                     'id': 1,
                                     'image_id': image_id,
                                     'iscrowd': 0}
                                    ]

        # Act
        actual_gt, actual_p = sut.to_coco(input, input)

        # Assert
        # Check ground truth
        self.assertEqual(len(actual_gt.getCatIds()), category_id, "Expected exactly one category")
        self.assertEqual(actual_gt.getImgIds(catIds=[category_id]), [image_id], "Exepected image id to match")
        self.assertEqual(actual_gt.loadAnns(actual_gt.getAnnIds(imgIds=[image_id])), expected_coco_annotation)
        self.assertEqual(actual_gt.getImgIds(), [image_id])
        # Check predictions
        self.assertEqual(actual_p.loadAnns(actual_p.getAnnIds(imgIds=[image_id])), expected_coco_annotation)

    def test_to_coco_single_item_diff_predict(self):
        """
        Test case to check a single image with single box, but different prediction
        :return:
        """
        # Arrange
        sut = CocoObjectDetectionApiAdapter()

        image_id = 99
        category_id = 1
        bbox = [1, 2, 10, 10]
        image1_anno = {"image_id": torch.tensor(image_id),
                       "boxes": torch.tensor([bbox]),
                       "labels": torch.tensor([category_id]),
                       "iscrowd": torch.tensor([0])}
        input_gt = [image1_anno]

        bbox_p = [100, 200, 200, 400]
        image_p_anno = {"image_id": torch.tensor(image_id),
                        "boxes": torch.tensor([bbox_p]),
                        "labels": torch.tensor([category_id]),
                        "iscrowd": torch.tensor([0])}
        input_p = [image_p_anno]

        expected_gt_coco_annotation = [{'area': 100,
                                        'bbox': bbox,
                                        'category_id': category_id,
                                        'id': 1,
                                        'image_id': image_id,
                                        'iscrowd': 0}
                                       ]

        expected_p_coco_annotation = [{'area': 80000,
                                       'bbox': bbox_p,
                                       'category_id': category_id,
                                       'id': 1,
                                       'image_id': image_id,
                                       'iscrowd': 0}
                                      ]

        # Act
        actual_gt, actual_p = sut.to_coco(input_gt, input_p)

        # Assert
        self.assertEqual(actual_gt.loadAnns(actual_gt.getAnnIds(imgIds=[image_id])), expected_gt_coco_annotation)
        # Check predictions
        self.assertEqual(actual_p.loadAnns(actual_p.getAnnIds(imgIds=[image_id])), expected_p_coco_annotation)

    def test_to_coco_two_item(self):
        """
        Test case single image with 2 boxes
        :return:
        """
        # Arrange
        sut = CocoObjectDetectionApiAdapter()

        image_id = 99
        category_id = 1
        bbox1 = [1, 2, 10, 10]
        bbox2 = [2, 10, 22, 22]
        image1_anno = {"image_id": torch.tensor(image_id),
                       "boxes": torch.tensor([bbox1, bbox2]),
                       "labels": torch.tensor([category_id, category_id]),
                       "iscrowd": torch.tensor([0, 1])}

        input = [image1_anno]

        expected_coco_annotation = [{'area': 100,
                                     'bbox': bbox1,
                                     'category_id': category_id,
                                     'id': 1,
                                     'image_id': image_id,
                                     'iscrowd': 0},
                                    {'area': 484,
                                     'bbox': bbox2,
                                     'category_id': category_id,
                                     'id': 2,
                                     'image_id': image_id,
                                     'iscrowd': 1}
                                    ]

        # Act
        actual_gt, actual_p = sut.to_coco(input, input)

        # Assert
        self.assertEqual(actual_gt.loadAnns(actual_gt.getAnnIds(imgIds=[image_id])), expected_coco_annotation)
        self.assertEqual(actual_p.loadAnns(actual_p.getAnnIds(imgIds=[image_id])), expected_coco_annotation)
