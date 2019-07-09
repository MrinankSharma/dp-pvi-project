import argparse
import ray

from sacred import Experiment
from sacred.observers import FileStorageObserver

from linreg.data import generate_random_dataset

ex = Experiment("Global DP-PVI Experiment")

parser = argparse.ArgumentParser(description="Global Linreg DP-PVI")
parser.add_argument("--config-file", type=str, help="JSON config file")


@ex.config
def default_config():
    """ Default Config: these settings can be overwritten!"""

    clipping = {
        "location": "worker",
        "type": "scaled_ppw",
        "value": 0.5
    }

    dataset = {
        "data_type": "homous",
        "whitened": True,
        "mean": {
            "type": "sample"
        },
        "model_noise_std": {
            "type": "sample",
            "min_val": 0.5,
            "max_val": 2,
        },
        "points_per_worker": {
            "type": "uniform",
            "value": 10
        },
        "prior_std": 5,
        "seeds_per_dataset": 1
    }
    experiment = {
        "datasets_per_experiment": 1,
        "num_workers": 20,
        "num_intervals": 250,
        "max_eps": 50,
        "experiment_tag": "default",
        "output_base_dir": "/Users/msharma/workspace/dp-pvi-project/logs/"
    }
    noise = {
        "location": "worker",
        "type": "scaled_worker",
        "value": 1
    }
    learning_rate = {
        "scheme": "constant",
        "start_value": 0.1
    }

    # adaptively set the file storage output directory
    ex.observers.append(
        FileStorageObserver.create(experiment["output_base_dir"] + "/" + experiment["experiment_tag"] + "/"))


@ex.automain
def global_linreg_experiment(dataset, experiment):
    ray.init()
    generate_random_dataset(dataset, experiment)



@ex.capture
def individual_linreg_experiment(_seed):
    pass
