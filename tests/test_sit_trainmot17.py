import os
import tempfile
from unittest import TestCase

import torch
from torch.utils.data import DataLoader

from dataset_factory_service_locator import DatasetFactoryServiceLocator
from model_factory_service_locator import ModelFactoryServiceLocator
from predict import Predict
from train_factory import TrainFactory


class TestSitTrainMot17(TestCase):

    def test_run(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "data", "small_clips")
        # get dataset
        dataset_factory = DatasetFactoryServiceLocator().get_factory("Mot17DetectionFactory")
        dataset = dataset_factory.get_dataset(img_dir)

        # get train factory
        model_factory_name = ModelFactoryServiceLocator().factory_names[0]
        train_factory = TrainFactory(model_factory_name, num_workers=1, epochs=1, batch_size=6, early_stopping=True,
                                     patience_epochs=2)
        output_dir = tempfile.mkdtemp()
        model_dir = tempfile.mkdtemp()

        # Act
        pipeline = train_factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir, model_dir)

    def test_run_predict(self):
        """
        Test case to make sure that the results emitted during training match inference
        :return:
        """
        # Arrange
        # get train factory
        model_factory_name = ModelFactoryServiceLocator().factory_names[0]

        # get datasetF
        img_dir = os.path.join(os.path.dirname(__file__), "data", "small_clips")

        # construct dataset
        dataset_factory = DatasetFactoryServiceLocator().get_factory("Mot17DetectionFactory")
        dataset = dataset_factory.get_dataset(img_dir)

        # kick off a single training run
        train_factory = TrainFactory(model_factory_name, num_workers=1, epochs=1, batch_size=6, early_stopping=True,
                                     patience_epochs=2)
        output_dir = tempfile.mkdtemp()
        model_dir = tempfile.mkdtemp()

        pipeline = train_factory.get(dataset)
        score, expected_predictions, model_path = pipeline.run(dataset, dataset, output_dir, model_dir)

        # construct predictor
        sut = Predict(model_dict_path=model_path, model_factory_name=model_factory_name,
                      num_classes=dataset.num_classes)

        # Create dataloader
        val_data_loader = DataLoader(dataset, num_workers=1, shuffle=False)

        # Act
        actual_predictions = sut.predict_batch(val_data_loader)

        # Assert
        for e, a in zip(expected_predictions, actual_predictions):
            for k in e:
                self.assertTrue(torch.all(torch.eq(e[k], a[k])),
                                "Could not match key {} , \n{}\n {}".format(k, e[k], a[k]))
