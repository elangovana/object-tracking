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

from torch.utils.data import DataLoader

from collate_functions import collate_fn


class TrainPipeline:

    def __init__(self, trainer, model, optimiser, loss_func=None, num_workers=None, batch_size=32):
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers is None:
            self.num_workers = 1 if os.cpu_count() == 1 else os.cpu_count() - 1
        self.trainer = trainer
        self.optimiser = optimiser
        self.loss_func = loss_func
        self.model = model

    def run(self, train_dataset, val_dataset, output_dir, model_dir, checkpoint_dir):
        train_data = DataLoader(train_dataset, num_workers=self.num_workers, shuffle=True, batch_size=self.batch_size,
                                collate_fn=collate_fn)
        val_data = DataLoader(val_dataset, num_workers=self.num_workers, shuffle=False, batch_size=self.batch_size,
                              collate_fn=collate_fn)

        return self.trainer.run(train_data, val_data, self.model, self.optimiser, output_dir, model_dir, checkpoint_dir)
