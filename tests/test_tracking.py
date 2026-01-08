import unittest
import tempfile
import shutil
import mlflow
from unittest.mock import MagicMock, patch
from src.tracking import MLflowTracker

class TestMLflowTracker(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for MLflow tracking to avoid persisting test data
        self.test_dir = tempfile.mkdtemp()
        mlflow.set_tracking_uri(f"file://{self.test_dir}")

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)
        mlflow.set_tracking_uri("") # Reset tracking URI

    @patch('mlflow.set_experiment')
    def test_init(self, mock_set_experiment):
        tracker = MLflowTracker(mode="train", experiment_name="test_exp")
        mock_set_experiment.assert_called_with("test_exp")
        self.assertEqual(tracker.mode, "train")

    def test_start_parent_run_train(self):
        tracker = MLflowTracker(mode="train", experiment_name="test_exp")
        with tracker.start_parent_run("run_1", params={"lr": 0.01}) as run:
            self.assertIsNotNone(run)
            self.assertEqual(run.info.run_name, "run_1")

    def test_start_trial_run_tune(self):
        tracker = MLflowTracker(mode="tune", experiment_name="test_exp")
        with tracker.start_parent_run("parent_run"):
            parent_run_id = mlflow.active_run().info.run_id
            with tracker.start_trial_run(trial_num=5, params={"hidden": 64}, parent_run_id=parent_run_id) as run:
                self.assertIsNotNone(run)
                self.assertEqual(run.info.run_name, "trial_5")

    @patch('mlflow.log_metrics')
    def test_log_epoch(self, mock_log_metrics):
        tracker = MLflowTracker(mode="train", experiment_name="test_exp")
        metrics = MagicMock()
        metrics.loss = 0.5
        metrics.accuracy = 0.8
        metrics.perfect_accuracy = 0.1
        metrics.ce_loss = 0.4
        metrics.degree_loss = 0.05
        metrics.crossing_loss = 0.05
        metrics.verify_loss = 0.0
        metrics.verify_balanced_acc = 0.0
        
        tracker.log_epoch(metrics, step=1, prefix="val_")
        mock_log_metrics.assert_called()
        args, kwargs = mock_log_metrics.call_args
        self.assertEqual(args[0]["val_loss"], 0.5)
        self.assertEqual(kwargs["step"], 1)

if __name__ == '__main__':
    unittest.main()
