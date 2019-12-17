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
from torch import nn
from torch.optim import Adam

from evaluators.map_evaluator import MAPEvaluator
from model_factory_service_locator import ModelFactoryServiceLocator
from train import Train
from train_pipeline import TrainPipeline


class TrainFactory:
    """
    Constructs the objects required to kick off training
    """

    def __init__(self, model_factory_name, epochs=50, early_stopping=True, patience_epochs=10, batch_size=32,
                 num_workers=None,
                 additional_args=None):

        self.model_factory_name = model_factory_name
        if num_workers is None and os.cpu_count() > 1:
            self.num_workers = min(4, os.cpu_count() - 1)
        else:
            self.num_workers = 0

        self.batch_size = batch_size
        self.patience_epochs = patience_epochs
        self.early_stopping = early_stopping
        self.epochs = epochs
        self.additional_args = additional_args or {}

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def _get_value(self, kwargs, key, default):
        value = kwargs.get(key, default)
        self.logger.info("Retrieving key {} with default {}, found {}".format(key, default, value))
        return value

    def get(self, train_dataset):
        accumulation_steps = int(self._get_value(self.additional_args, "accumulation_steps", "1"))
        self.logger.info("Using accumulation steps {}".format(accumulation_steps))
        evaluator = MAPEvaluator()
        trainer = Train(patience_epochs=self.patience_epochs, early_stopping=self.early_stopping,
                        epochs=self.epochs, evaluator=evaluator, accumulation_steps=accumulation_steps)

        # Model
        self.logger.info("Using model {}".format(self.model_factory_name))
        model_factory = ModelFactoryServiceLocator().get_factory(self.model_factory_name)
        model = model_factory.get_model(num_classes=train_dataset.num_classes)
        # TODO: Enable multi gpu, nn.dataparallel doesnt really work...
        if torch.cuda.device_count() > 1:
            self.logger.info("Using nn.DataParallel../ multigpu.. Currently not working..")
            model = nn.DataParallel(model)
            # Increase batch size so that is equivalent to the batch
            self.batch_size = self.batch_size * torch.cuda.device_count()
        self.logger.info("Using model {}".format(type(model)))

        # Define optimiser
        learning_rate = float(self._get_value(self.additional_args, "learning_rate", ".0001"))
        self.logger.info("Using learning_rate {}".format(learning_rate))

        # weight_decay = float(self._get_value(self.additional_args, "weight_decay", "5e-5"))
        # momentum = float(self._get_value(self.additional_args, "momentum", ".9"))
        # optimiser = SGD(lr=learning_rate, params=model.parameters(), momentum=momentum, weight_decay=weight_decay)
        optimiser = Adam(lr=learning_rate, params=model.parameters())
        self.logger.info("Using optimiser {}".format(type(optimiser)))

        # Kick off training pipeline
        train_pipeline = TrainPipeline(batch_size=self.batch_size,
                                       optimiser=optimiser,
                                       trainer=trainer,
                                       num_workers=self.num_workers,
                                       model=model)

        return train_pipeline
