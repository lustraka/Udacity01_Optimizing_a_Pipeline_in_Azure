# Create a custom environment that encapsulates the training script's dependencies
# (1) Define your conda dependencies in a YAML file
# (2) Create an Azure ML environment from this Conda environment specification.

# (1) ------------------------------------------------------------------------------------
%%writefile conda_dependencies.yml

dependencies:
- python=3.6.2
- pip=20.2.4
- pip:
    - azureml-defaults
    - scikit-learn

# (2) ------------------------------------------------------------------------------------
from azureml.core import Environment

sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

# ------------------------------------------------------------------------------------
# TODO: Create compute cluster
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)

# Delete() is used to deprovision and delete the AmlCompute target. Useful if you want to re-use the compute name 
# 'cpu-cluster' in this case but use a different VM family for instance.

# cpu_cluster.delete()

# ------------------------------------------------------------------------------------
# 6. Automated ML and Hzperparameter Tunning > 2. Hyperparameter Tuning with HyperDrive
# Controlling HyperDrive with the SDK
from azureml.train.hyperdrive import BayesianParameterSampling
from azureml.train.hyperdrive import uniform, choice
param_sampling = BayesianParameterSampling( {
        "learning_rate": uniform(0.05, 0.1),
        "batch_size": choice(16, 32, 64, 128)
    }
)

# ------------------------------------------------------------------------------------
# And finally, we will need to find the best model parameters. Here's an example:

best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()['runDefinition']['Arguments']

print('Best Run Id: ', best_run.id)
print('\n Accuracy:', best_run_metrics['accuracy'])
print('\n learning rate:', parameter_values[3])
print('\n keep probability:', parameter_values[5])
print('\n batch size:', parameter_values[7])

# ------------------------------------------------------------------------------------
# 6. Automated ML and Hyperparameter Tunning > 3. Exercise: Hyperparameter Tuning with HyperDrive
# My turn!
from azureml.widgets import RunDetails
# from azureml.train.sklearn import SKLearn >> DEPRECATED. Use the ScriptRunConfig object with your own defined environment.
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
import os

# Specify parameter sampler
ps = RandomParameterSampling(
    {
    'C': uniform(0.001, 1.0), # For regularization
    'max_iter': choice(10, 50, 100) # Max number of epochs
    }
)

# Specify a Policy
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

if "training" not in os.listdir():
    os.mkdir("./training")

# DEPRECATED. Create a SKLearn estimator for use with train.py
# Create a ScriptRunConfig object to specify the configuration details of your training job

from azureml.core import ScriptRunConfig

est = ScriptRunConfig(source_directory='.',
                      script='train.py',
                      compute_target=cpu_cluster,
                      environment=sklearn_env)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = ### YOUR CODE HERE ###
