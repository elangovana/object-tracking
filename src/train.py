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

    def __init__(self, evaluator, device=None, snapshotter=None, early_stopping=True, patience_epochs=10, epochs=10,
                 results_writer=None):
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

    def run(self, train_data, val_data, model, loss_func, optimiser, output_dir):
        self.logger.info("Running training...")

        model.to(device=self.device)
        best_loss = None
        best_score = None
        patience = 0

        result_logs = []

        for e in range(self.epochs):

            for i, (p_x, q_x, n_x, target) in enumerate(train_data):
                # Set up train mode
                model.train()

                # Copy to device
                p_x = p_x.to(device=self.device)
                q_x = q_x.to(device=self.device)
                n_x = n_x.to(device=self.device)

                target = target.to(device=self.device)

                # Forward pass
                p_features = model(p_x)
                q_features = model(q_x)
                n_features = model(n_x)
                self.logger.debug("Computing loss function: ")
                loss = loss_func(p_features, q_features, n_features, target)
                self.logger.debug("Computing loss function complete ")

                # Backward
                optimiser.zero_grad()
                loss.backward()

                # Update weights
                optimiser.step()

                self.logger.debug(
                    "Batch {}/{}, loss {}".format(i, e, loss.item()))

            train_loss, train_loss_mean, train_loss_std, train_score = self._compute_validation_loss(train_data, model,
                                                                                                     loss_func)

            # Validation loss
            val_loss, val_loss_mean, val_loss_std, val_score = self._compute_validation_loss(val_data, model,
                                                                                             loss_func)

            # Save snapshots
            if best_score is None or val_score > best_score:
                self.logger.info(
                    "Snapshotting as current score {} is > previous best {}".format(val_score, best_score))
                self.snapshotter.save(model, output_dir=output_dir, prefix="snapshot_")
                best_score = val_score
                best_loss = val_loss
                # Reset patience if loss decreases
                patience = 0
            # score is the same but lower loss
            elif val_score == best_score and best_loss is not None and val_loss < best_loss:
                self.logger.info(
                    "Snapshotting as current loss {} is < previous best {} for score {}".format(val_loss, best_loss,
                                                                                                val_score))
                self.snapshotter.save(model, output_dir=output_dir, prefix="snapshot_")
                best_loss = val_loss
                # Reset patience if loss decreases
                patience = 0
            else:
                # No increase in best loss so increase patience counter
                patience += 1

            print("###score: train_loss### {}".format(train_loss))
            print("###score: val_loss### {}".format(val_loss))
            print("###score: val_loss_std### {}".format(val_loss_std))
            print("###score: train_loss_std### {}".format(train_loss_std))
            print("###score: train_score### {}".format(train_score))
            print("###score: val_score### {}".format(val_score))

            # print and store run logs
            self.logger.info(
                "epoch: {}, train_loss {}, val_loss {}, val_loss_mean {}, train_loss_mean {}, val_loss_std {}, train_loss_std {}, train_score {}, val_score {}".format(
                    e, train_loss,
                    val_loss,
                    val_loss_mean, train_loss_mean, val_loss_std, train_loss_std,
                    train_score,
                    val_score))
            result_logs.append([e, train_loss,
                                val_loss,
                                val_loss_mean, train_loss_mean, val_loss_std, train_loss_std,
                                train_score,
                                val_score])

            # Early stopping
            if self.early_stopping and patience > self.patience_epochs:
                self.logger.info("No decrease in loss for {} epochs and hence stopping".format(self.patience_epochs))
                break
            else:
                self.logger.info("Patience is {}".format(patience))

        self.logger.info("The best val score is {}".format(best_score))
        self.results_writer.dump_object(result_logs, output_dir, "epochs_loss")

    def _compute_validation_loss(self, data, model, loss_func):
        # Model Eval mode
        model.eval()

        losses = []

        # No grad
        predictions = []
        target_items = []
        with torch.no_grad():
            for i, (p_x, q_x, n_x, target) in enumerate(data):
                target_items.append(target)

                # Copy to device
                p_x = p_x.to(device=self.device)
                q_x = q_x.to(device=self.device)
                n_x = n_x.to(device=self.device)

                target = target.to(device=self.device)

                # Forward pass
                p_features = model(p_x)
                q_features = model(q_x)
                n_features = model(n_x)
                self.logger.debug("Computing loss function: ")
                val_loss = loss_func(p_features, q_features, n_features, target)

                # Check for Nans
                if q_features.ne(q_features).any():

                    if q_features.ne(q_features).all():
                        self.logger.warning(
                            "All outputs are NaNs in predicted features in batch {}.. This could be because of exploding or vanishing gradients".format(
                                i))
                    else:
                        self.logger.warning(
                            "There are NaNs in predicted features in batch {}.. This could be because of exploding or vanishing gradients".format(
                                i))

                # Total loss
                losses.append(val_loss.item())

                # Score to get score
                predictions.append(q_features)

        predictions = torch.cat(predictions, dim=0)
        target_items = torch.cat(target_items, dim=0)
        score = self._get_score(predictions, target_items)
        losses = torch.Tensor(losses)

        return losses.sum().item(), losses.mean().item(), losses.std().item(), score

    def _get_score(self, predicted, target):
        return self.evaluator(predicted, target)
