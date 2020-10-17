from azureml.core import Run
from azureml.core.compute import AksCompute, ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.experiment import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.workspace import Workspace
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, PublishedPipeline, PipelineRun, Schedule, TrainingOutput
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.train.automl import AutoMLConfig, AutoMLStep

import azureml
import os

experiment_name = 'category-based-propensity'
cluster_name = 'wba-cluster'

PRODUCT_CATEGORIES = 10

# Data functions
def get_cat(index, lower=False):
    if index == 0:
        cat = 'FNN'
        cat = cat if not lower else cat.lower()
    elif index == 1:
        cat = 'WLN'
        cat = cat if not lower else cat.lower()
    else:
        cat = index + 1
    
    return cat

def prep_data_file(index):
    with open('get_data.py') as f:
        content = f.read()

    cat = get_cat(index)
    content = content.replace('BOUGHT_CATEGORY_FNN', 'BOUGHT_CATEGORY_{}'.format(cat))
    
    cat = get_cat(index, lower=True)
    with open('get_data_c_{}.py'.format(cat), 'w') as f:
        f.write(content)

for i in range(PRODUCT_CATEGORIES):
    prep_data_file(i)

# Connect to Azure Machine Learning
try:
    ws = Workspace.from_config()
except:
    ws = Workspace(subscription_id = os.environ['SUBSCRIPTIONID'],
                   resource_group = os.environ['RESOURCEGROUP'],
                   workspace_name = os.environ['WORKSPACENAME'])
    ws.write_config()
    
    print('Workspace config file written')

# Provision a compute target
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           min_nodes=1,
                                                           max_nodes=12)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Prepare experiment
ds = ws.get_default_datastore()
experiment = Experiment(ws, experiment_name)

# Create our experiment configuration
input_data = DataReference(datastore=ds, 
                           data_reference_name='training_data',
                           path_on_datastore='boots',
                           mode='download',
                           path_on_compute='/tmp/azureml_runs',
                           overwrite=True)

run_config = RunConfiguration(framework="python")
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE
run_config.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=['azureml-sdk[automl]'], conda_packages=['numpy','py-xgboost<=0.80'])

# Build a pipeline
steps = []
current = None

# Build a model for every category
for i in range(PRODUCT_CATEGORIES):
    cat = get_cat(i, lower=True)

    # These are the two outputs from AutoML
    metrics_data = PipelineData(name='metrics_data__category_{}'.format(cat),
                                datastore=ds,
                                pipeline_output_name='metrics_output__category_{}'.format(cat),
                                training_output=TrainingOutput(type='Metrics'))

    model_data = PipelineData(name='model_data__category_{}'.format(cat),
                              datastore=ds,
                              pipeline_output_name='best_model_output__category_{}'.format(cat),
                              training_output=TrainingOutput(type='Model'))

    # AutoML config (note different data files for each model so it's not shared)
    automl_config = AutoMLConfig(task = 'classification',
                                 iterations = 25,
                                 iteration_timeout_minutes = 5, 
                                 max_cores_per_iteration = 2,
                                 max_concurrent_iterations = 8,
                                 primary_metric = 'accuracy',
                                 data_script = 'get_data_c_{}.py'.format(cat),
                                 run_configuration = run_config,
                                 compute_target = compute_target,
                                 path = '.',
                                 n_cross_validations = 2,
                                 preprocess = True,)
    
    # AutoML action
    automl_step = AutoMLStep(name='automl_module__category_{}'.format(cat),
                             automl_config=automl_config,
                             inputs=[input_data],
                             outputs=[metrics_data, model_data],
                             allow_reuse=False)
    
    # Custom script action to register the model afterwards
    register_step = PythonScriptStep(name='register__category_{}'.format(cat),
                                     script_name='register.py',
                                     compute_target=compute_target,
                                     source_directory='.',
                                     arguments=['--model_name', 'category_{}_model.pkl'.format(cat), '--model_path', model_data],
                                     inputs=[model_data],
                                     allow_reuse=False)
    
    # And chain them together so they run sequentially
    if current:
        automl_step.run_after(current)

    current = register_step

    steps.append(automl_step)
    steps.append(register_step)

pipeline = Pipeline(description='Generate recommendation models',
                    workspace=ws,
                    steps=steps)

pipeline.validate()

# Once published, we can invoke on demand via the SDK or via a REST endpoint
published_pipeline = pipeline.publish(name='category-based-propensity-pipeline')

# Run it on demand
published_pipeline.submit(ws, published_pipeline.name)