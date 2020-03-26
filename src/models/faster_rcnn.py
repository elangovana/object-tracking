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
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn


class FasterRCnn(nn.Module):

    def __init__(self, num_classes, rpn_pre_nms_top_n_train=100, rpn_pre_nms_top_n_test=100,
                 rpn_post_nms_top_n_train=100, rpn_post_nms_top_n_test=100, rpn_anchor_generator=None):
        super().__init__()
        # load a model pre-trained pre-trained on COCO
        self.model = fasterrcnn_resnet50_fpn(pretrained=True, rpn_pre_nms_top_n_train=100, rpn_pre_nms_top_n_test=100,
                                             rpn_post_nms_top_n_train=100, rpn_post_nms_top_n_test=100,
                                             rpn_anchor_generator=rpn_anchor_generator)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, *input):
        return self.model(*input)
