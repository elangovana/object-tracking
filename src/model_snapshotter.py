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
import logging
import os

import torch


class Snapshotter(object):
    """
    Takes a model snapshot
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def save(self, model, output_dir, prefix="Snapshot"):
        snapshot_prefix = os.path.join(output_dir, prefix)
        snapshot_path = snapshot_prefix + 'model.pt'

        self.logger.info("Snappshotting model to {} ".format(snapshot_path))

        torch.save(model, snapshot_path)
