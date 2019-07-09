import argparse

from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create(db_name='DP_PVI'))
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
        "dataset": "toy_1d",
        "data_type": "homous",
        "mean": "sample",
        "model_noise_std": "sample",
        "points_per_worker": 10,
        "prior_std": 5,
        "seeds_per_dataset": 1
    }
    experiment = {
        "datasets_per_experiment": 1,
        "num_workers": 20,
        "num_intervals": 250,
        "max_eps": 50,
        "experiment_tag": "default",
        "output_base_dir": ""
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


@ex.automain
def global_linreg_experiment(clipping):
    print(clipping)
    ex.log_scalar("testing.value", 1)
