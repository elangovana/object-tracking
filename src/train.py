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

import torch

from model_snapshotter import Snapshotter
from result_writer import ResultWriter


class Train:

    def __init__(self, device=None, snapshotter=None, early_stopping=True, patience_epochs=10, epochs=10,
                 results_writer=None, evaluator=None):
        # TODO: currently only single GPU
        self.evaluator = evaluator
        self.results_writer = results_writer
        self.epochs = epochs
        self.patience_epochs = patience_epochs
        self.early_stopping = early_stopping
        self.snapshotter = snapshotter
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def snapshotter(self):
        self._snapshotter = self._snapshotter or Snapshotter()
        return self._snapshotter

    @snapshotter.setter
    def snapshotter(self, value):
        self._snapshotter = value

    @property
    def results_writer(self):
        self._results_writer = self._results_writer or ResultWriter()
        return self._results_writer

    @results_writer.setter
    def results_writer(self, value):
        self._results_writer = value

    def run(self, train_data, val_data, model, optimiser, output_dir):
        self.logger.info("Running training...")

        model.to(device=self.device)
        best_score = None
        patience = 0

        result_logs = []

        for e in range(self.epochs):
            total_train_loss = 0
            for i, (images, targets) in enumerate(train_data):
                # Set up train mode
                model.train()

                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)

                self.logger.debug("Computing loss function complete ")

                # Backward
                optimiser.zero_grad()
                loss = sum(loss for loss in loss_dict.values())
                loss.backward()

                # Update weights
                optimiser.step()

                self.logger.debug(
                    "Batch {}/{}, loss {}".format(i, e, loss.item()))

                total_train_loss += loss.item()

            # compute train score
            train_target, train_predictions, train_score = self._compute_validation_loss(train_data, model)

            # Validation score
            val_target, val_predictions, val_score = self._compute_validation_loss(val_data, model)

            # Save snapshots
            if best_score is None or val_score > best_score:
                self.logger.info(
                    "Snapshotting as current score {} is > previous best {}".format(val_score, best_score))
                self.snapshotter.save(model, output_dir=output_dir, prefix="snapshot_")
                best_score = val_score

                patience = 0

            else:
                patience += 1

            print("###score: train_loss### {}".format(total_train_loss))
            print("###score: train_score### {}".format(train_score))
            print("###score: val_score### {}".format(val_score))

            self.logger.info(
                "epoch: {}, train_loss {}, train_score {}, val_score {}".format(e, total_train_loss, train_score,
                                                                                val_score))
            result_logs.append([e, train_score, val_score])

            # Early stopping
            if self.early_stopping and patience > self.patience_epochs:
                self.logger.info("No decrease in loss for {} epochs and hence stopping".format(self.patience_epochs))
                break
            else:
                self.logger.info("Patience is {}".format(patience))

        self.logger.info("The best val score is {}".format(best_score))
        self.results_writer.dump_object(result_logs, output_dir, "epochs_loss")

    def _compute_validation_loss(self, data, model):
        # Model Eval mode
        model.eval()

        predictions = []
        target_items = []

        # No grad
        with torch.no_grad():
            for i, (images, targets) in enumerate(data):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                # in eval model only gets the predictions and not loss..
                predicted_batch = model(images, targets)

                self.logger.debug("Computing loss function: ")

                predictions.extend(predicted_batch)
                target_items.extend(targets)

        score = self._get_score(target_items, predictions)

        return predictions, target_items, score

    def _get_score(self, target, predicted):
        return self.evaluator(target, predicted)
