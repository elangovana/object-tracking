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
import torch

from models.base_model_factory import BaseModelFactory
from models.faster_rcnn import FasterRCnn
import logging


class FasterRcnnFactory(BaseModelFactory):

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def load_model(self, model_path, num_classes, **kwargs):
        rpn_pre_nms_top_n_train = int(self._get_value(kwargs, "rpn_pre_nms_top_n_train", "100"))
        rpn_pre_nms_top_n_test = int(self._get_value(kwargs, "rpn_pre_nms_top_n_test", "100")),
        rpn_post_nms_top_n_train = int(self._get_value(kwargs, "rpn_post_nms_top_n_train", "100"))
        rpn_post_nms_top_n_test = int(self._get_value(kwargs, "rpn_post_nms_top_n_test", "100"))

        model = FasterRCnn(num_classes, rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model

    def get_model(self, num_classes, **kwargs):
        rpn_pre_nms_top_n_train = int(self._get_value(kwargs, "rpn_pre_nms_top_n_train", "100"))
        rpn_pre_nms_top_n_test = int(self._get_value(kwargs, "rpn_pre_nms_top_n_test", "100")),
        rpn_post_nms_top_n_train = int(self._get_value(kwargs, "rpn_post_nms_top_n_train", "100"))
        rpn_post_nms_top_n_test = int(self._get_value(kwargs, "rpn_post_nms_top_n_test", "100"))

        model = FasterRCnn(num_classes, rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                           rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                           rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                           rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)
        return FasterRCnn(num_classes, rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
                          rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
                          rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
                          rpn_post_nms_top_n_test=rpn_post_nms_top_n_test)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value
