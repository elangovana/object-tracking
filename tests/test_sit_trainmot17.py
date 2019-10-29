import os
import tempfile
from unittest import TestCase

from dataset_factory_service_locator import DatasetFactoryServiceLocator
from train_factory import TrainFactory


class TestSitTrainMot17(TestCase):

    def test_run(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "data", "small_clips")
        # get dataset
        dataset_factory = DatasetFactoryServiceLocator().get_factory("Mot17DetectionFactory")
        dataset = dataset_factory.get_dataset(img_dir)

        # get train factory
        train_factory = TrainFactory(num_workers=1, epochs=1, batch_size=6, early_stopping=True, patience_epochs=2)
        output_dir = tempfile.mkdtemp()

        # Act
        pipeline = train_factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir)
