from azureml.core import Workspace, Experiment

ws = Workspace.from_config()
exp = Experiment(workspace=ws, name="udacity-project")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()

# Create compute cluster
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

# Create a custom environment that encapsulates the training script's dependencies
# (1) Define your conda dependencies in a YAML file
# (2) Create an Azure ML environment from this Conda environment specification.

%%writefile conda_dependencies.yml

dependencies:
- python=3.8.1
- pip=20.1.1
- pip:
    - azureml-defaults
    - scikit-learn
from azureml.core import Environment

sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

from azureml.widgets import RunDetails
# from azureml.train.sklearn import SKLearn >> DEPRECATED. Use the ScriptRunConfig object with your own defined environment.
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform, choice
import os

# Specify parameter sampler
ps = RandomParameterSampling(
    {
    'C': uniform(0.01, 100.0), # For regularization
    'max_iter': choice(500, 1000, 2000) # Max number of epochs
    }
)

# Specify a Policy
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

if "training" not in os.listdir():
    os.makedirs("training", exist_ok=True)

# !Create a SKLearn estimator is DEPRECATED > Use:
# Create a ScriptRunConfig object to specify the configuration details of your training job

src = ScriptRunConfig(source_directory='.',
                      script='train.py',
                      compute_target=cpu_cluster,
                      environment=sklearn_env)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(run_config=src,
                                    hyperparameter_sampling=ps,
                                    policy=policy,
                                    primary_metric_name='Accuracy',
                                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                    max_total_runs=20,
                                    max_concurrent_runs=4,
                                    max_duration_minutes=30)


# Submit your hyperdrive run to the experiment and show run details with the widget.
hdr = exp.submit(config=hyperdrive_config)
RunDetails(hdr).show()
hdr.wait_for_completion(show_output=True)

# Get your best run and save the model from that run.
best_run = hdr.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()

print('Best Run Id: ', best_run.id)
print('Regularization Strength:', best_run_metrics['Regularization Strength:'])
print('Max iterations:', best_run_metrics['Max iterations:'])
print('Accuracy:', best_run_metrics['Accuracy'])

best_run.get_file_names()

best_run.download_file(best_run.get_file_names()[-1], output_file_path='./outputs/')

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

#model = run.register_model(model_name='sklearn-logistic-regression', model_path="./outputs/model.joblib") # causes ModelPathNotFoundException
model = Model.register(ws, model_path='outputs/model.joblib', model_name='sklearn-logistic-regression', tags=best_run_metrics)

print(model.name, model.id, model.version, sep='\t')

#########################################################
from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
url_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
ds = TabularDatasetFactory.from_delimited_files(url_path)

from train import get_X_y

# Use the clean_data function to clean your data.
x, y = get_X_y(ds)

from azureml.train.automl import AutoMLConfig

# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.
automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task=,
    primary_metric=,
    training_data=,
    label_column_name=,
    n_cross_validations=)

# Submit your automl run

### YOUR CODE HERE ###

# Retrieve and save your best automl model.

### YOUR CODE HERE ###

# Delete() is used to deprovision and delete the AmlCompute target. 

cpu_cluster.delete()



