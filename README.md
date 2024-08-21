# INSTALLATION

See the [installation page](docs/installation.md) in the `docs` folder.

# OVERVIEW

The gymnasiums in this repository are each made of three files (i.e., 3 classes):

* A class for the system (e.g., the robot).
* A class for the environment.
* A Gymnasium class that simulates the system in its environment.

The files for the gymnasiums are in [ai4rgym/envs](ai4rgym/envs)

Currently, only one gymnasium is available:

* Autonomous Driving Gymnasium ([documentation here](docs/ad_gym.md))

The other key folders in this repository are:

* `evaluation`  :  This folder contain utility functions that can be used for evaluating the performance of a policy. The main types of function provided are for performing simulations and plotting. ([Documentation here for the evaluation utilities](docs/evaluation_utilities.md))

* `policies`  :  This folder contains classes of different types of policies that conform to the requirement of the simulation utilities. ([Documentation here for the policy classes](docs/policy_classes.md))

* `unit_test` :  This folder contains utility functions that can be used to test the validity of various aspects of the systems, environments, and gymnasiums. ([Documentation here for the unit testing utilities](docs/unit_test_utilities.md))

* `examples`  :  This folder contain examples python scripts for using the various gymnasiums, classes, and utility functions above. ([Documentation here for the example script](docs/examples.md))