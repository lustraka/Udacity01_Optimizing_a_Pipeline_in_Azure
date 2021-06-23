# Notes 2021-06-23
## Details of a HyperDrive model
`{'runId': 'HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5',
 'target': 'cpucluster',
 'status': 'Completed',
 'startTimeUtc': '2021-06-23T13:08:03.554571Z',
 'endTimeUtc': '2021-06-23T13:08:42.828859Z',
 'properties': {'_azureml.ComputeTargetType': 'amlcompute',
  'ContentSnapshotId': '987fcb30-e7ff-4b23-a10b-b90961bcd1c6',
  'ProcessInfoFile': 'azureml-logs/process_info.json',
  'ProcessStatusFile': 'azureml-logs/process_status.json'},
 'inputDatasets': [],
 'outputDatasets': [],
 'runDefinition': {'script': 'train.py',
  'command': '',
  'useAbsolutePath': False,
  'arguments': ['--C', '70.06612587346416', '--max_iter', '1000'],
  'sourceDirectoryDataStore': None,
  'framework': 'Python',
  'communicator': 'None',
  'target': 'cpucluster',
  'dataReferences': {},
  'data': {},
  'outputData': {},
  'datacaches': [],
  'jobName': None,
  'maxRunDurationSeconds': 2592000,
  'nodeCount': 1,
  'priority': None,
  'credentialPassthrough': False,
  'identity': None,
  'environment': {'name': 'sklearn-env',
   'version': 'Autosave_2021-06-23T12:41:56Z_625b1523',
   'python': {'interpreterPath': 'python',
    'userManagedDependencies': False,
    'condaDependencies': {'dependencies': ['python=3.8.1',
      'pip=20.1.1',
      {'pip': ['azureml-defaults', 'scikit-learn']}],
     'name': 'azureml_55adbc95158c5e67a1b4ea051b715640'},
    'baseCondaEnvironment': None},
   'environmentVariables': {'EXAMPLE_ENV_VAR': 'EXAMPLE_VALUE'},
   'docker': {'baseImage': 'mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210513.v1',
    'platform': {'os': 'Linux', 'architecture': 'amd64'},
    'baseDockerfile': None,
    'baseImageRegistry': {'address': None, 'username': None, 'password': None},
    'enabled': False,
    'arguments': []},
   'spark': {'repositories': [], 'packages': [], 'precachePackages': True},
   'inferencingStackVersion': None},
  'history': {'outputCollection': True,
   'directoriesToWatch': ['logs'],
   'enableMLflowTracking': True,
   'snapshotProject': True},
  'spark': {'configuration': {'spark.app.name': 'Azure ML Experiment',
    'spark.yarn.maxAppAttempts': '1'}},
  'parallelTask': {'maxRetriesPerWorker': 0,
   'workerCountPerNode': 1,
   'terminalExitCodes': None,
   'configuration': {}},
  'amlCompute': {'name': None,
   'vmSize': None,
   'retainCluster': False,
   'clusterMaxNodeCount': None},
  'aiSuperComputer': {'instanceType': None,
   'imageVersion': None,
   'location': None,
   'aiSuperComputerStorageData': None,
   'interactive': False,
   'scalePolicy': None,
   'virtualClusterArmId': None,
   'tensorboardLogDirectory': None,
   'sshPublicKey': None},
  'tensorflow': {'workerCount': 1, 'parameterServerCount': 1},
  'mpi': {'processCountPerNode': 1},
  'pyTorch': {'communicationBackend': 'nccl', 'processCount': None},
  'hdi': {'yarnDeployMode': 'Cluster'},
  'containerInstance': {'region': None, 'cpuCores': 2.0, 'memoryGb': 3.5},
  'exposedPorts': None,
  'docker': {'useDocker': False,
   'sharedVolumes': True,
   'shmSize': '2g',
   'arguments': []},
  'cmk8sCompute': {'configuration': {}},
  'commandReturnCodeConfig': {'returnCode': 'Zero',
   'successfulReturnCodes': []},
  'environmentVariables': {},
  'applicationEndpoints': {}},
 'logFiles': {'azureml-logs/55_azureml-execution-tvmps_0f9fa10f6ad120da918b6d393b4444cd5dc95bf8dcf4c74b2e565d0ddc3a39a1_d.txt': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/azureml-logs/55_azureml-execution-tvmps_0f9fa10f6ad120da918b6d393b4444cd5dc95bf8dcf4c74b2e565d0ddc3a39a1_d.txt?sv=2019-02-02&sr=b&sig=yY4ZTnANnVnnDtZlHaMtjM1gIPFCdm7%2FludVSEIypUs%3D&st=2021-06-23T13%3A05%3A48Z&se=2021-06-23T21%3A15%3A48Z&sp=r',
  'azureml-logs/65_job_prep-tvmps_0f9fa10f6ad120da918b6d393b4444cd5dc95bf8dcf4c74b2e565d0ddc3a39a1_d.txt': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/azureml-logs/65_job_prep-tvmps_0f9fa10f6ad120da918b6d393b4444cd5dc95bf8dcf4c74b2e565d0ddc3a39a1_d.txt?sv=2019-02-02&sr=b&sig=BqW2M1%2BLjtAVunViWDK53p1baD5HHhod28pvJib6QIw%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'azureml-logs/70_driver_log.txt': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/azureml-logs/70_driver_log.txt?sv=2019-02-02&sr=b&sig=0AZe7q0%2BoB30Cyee6WlHdI7Q5IS5AEh%2BTyyojBZ35a8%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'azureml-logs/75_job_post-tvmps_0f9fa10f6ad120da918b6d393b4444cd5dc95bf8dcf4c74b2e565d0ddc3a39a1_d.txt': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/azureml-logs/75_job_post-tvmps_0f9fa10f6ad120da918b6d393b4444cd5dc95bf8dcf4c74b2e565d0ddc3a39a1_d.txt?sv=2019-02-02&sr=b&sig=dODjBDT0T1tnom6GItyqmfPe6hlMVl4XOLQkfUtQ0g8%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'azureml-logs/process_info.json': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/azureml-logs/process_info.json?sv=2019-02-02&sr=b&sig=IFhsUFB8JkxuG%2FBJLEwmDENbhKo9H1F8d7nKpSntGEo%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'azureml-logs/process_status.json': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/azureml-logs/process_status.json?sv=2019-02-02&sr=b&sig=S7o2Q8spY9ojowyqHrawkM%2FlpjG68AM02We8IW%2FNNV0%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'logs/azureml/93_azureml.log': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/logs/azureml/93_azureml.log?sv=2019-02-02&sr=b&sig=GQ7Ky3nhE2gLJOxxMELZ5H1a5giLOfnErI00Q2X1tF8%3D&st=2021-06-23T13%3A05%3A48Z&se=2021-06-23T21%3A15%3A48Z&sp=r',
  'logs/azureml/dataprep/backgroundProcess.log': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/logs/azureml/dataprep/backgroundProcess.log?sv=2019-02-02&sr=b&sig=1FY2TQHympPsEdoeAzni92S%2FHd6A8ZdanQKXvW8pO90%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'logs/azureml/dataprep/backgroundProcess_Telemetry.log': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/logs/azureml/dataprep/backgroundProcess_Telemetry.log?sv=2019-02-02&sr=b&sig=8wDPnxe5DI62vrlovw9hKjuhF9fP4qx2l6aWi4YTc%2Fg%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'logs/azureml/job_prep_azureml.log': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/logs/azureml/job_prep_azureml.log?sv=2019-02-02&sr=b&sig=GB52qHDYcYkoNOsLr23W22bGLIYRJK6NqpFdY5std4E%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r',
  'logs/azureml/job_release_azureml.log': 'https://mlstrg147788.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5/logs/azureml/job_release_azureml.log?sv=2019-02-02&sr=b&sig=4qXh%2Fue%2FTJtRRZ2rp76RWGqW8jNRFyWNti2Bg2MSH7I%3D&st=2021-06-23T13%3A05%3A49Z&se=2021-06-23T21%3A15%3A49Z&sp=r'},
 'submittedBy': 'ODL_User 147788'}`
+ Best Run Id:  HD_cad5df1a-3e63-40b9-82de-7100cc5be99b_5
+ Regularization Strength: 70.06612587346416
+ Max iterations: 1000
+ Accuracy: 0.8982762806506434

## Details of a AutoML model
**Data transformation**
{
    "class_name": "MaxAbsScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {},
    "prepared_kwargs": {},
    "spec_class": "preproc"
}

**Training algorithm**
{
    "class_name": "LightGBMClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "boosting_type": "goss",
        "colsample_bytree": 0.99,
        "learning_rate": 0.08947473684210526,
        "max_bin": 60,
        "max_depth": 3,
        "min_child_weight": 1,
        "min_data_in_leaf": 0.010353793103448278,
        "min_split_gain": 0.3684210526315789,
        "n_estimators": 400,
        "num_leaves": 92,
        "reg_alpha": 0.21052631578947367,
        "reg_lambda": 0.9473684210526315,
        "subsample": 1
    },
    "prepared_kwargs": {},
    "spec_class": "sklearn"
}
## ConfigException: ConfigException
ConfigException: ConfigException:
	Message: Input of type '<class 'pandas.core.frame.DataFrame'>' is not supported. Supported types: [azureml.data.tabular_dataset.TabularDataset]
	InnerException: None
	ErrorResponse 
{
    "error": {
        "code": "UserError",
        "message": "Input of type '<class 'pandas.core.frame.DataFrame'>' is not supported. Supported types: [azureml.data.tabular_dataset.TabularDataset]",
        "details_uri": "https://aka.ms/AutoMLConfig",
        "target": "training_data",
        "inner_error": {
            "code": "BadArgument",
            "inner_error": {
                "code": "ArgumentInvalid",
                "inner_error": {
                    "code": "InvalidInputDatatype"
                }
            }
        }
    }
}
