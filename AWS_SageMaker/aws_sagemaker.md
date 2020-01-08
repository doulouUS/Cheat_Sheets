# Set up


# Tensorflow 

## Script mode

Use Tensorflow eager mode: shift from the previous graph execution mode previously used. Natural direction of Tensorflow (adopted in v2).

Advantages:
* More intuitive
* Easier to debug
* Contrary to graph mode which keeps objects as long as `tf.Session()` is live, objects' lifetime is determined by their respective Python objects
* `tf.keras` code can be used as such

Workflow:
1. Script in a `__main__` guard, with all input arguments as command line inputs (typically data paths and model hyperparameters)
	- parse command line arguments
	- load data
	- build model
	- train 
	- save model to S3  (not needed as performed automatically in step 2 ?)

	This script will be called in a container by Amazon SageMaker in the next step, and is teared down after execution, hence the saving step

	ex: call this script `train.py`

2. Launch training SageMaker's Tensorflow wrapper
```
import sagemaker
from sagemaker.tensorflow import TensorFlow

model_dir = '/opt/ml/model'
train_instance_type = 'local'  # or 'ml.c4.xlarge', see next parts
hyperparameters = {'epochs': 10, 'batch_size': 128}


inputs = {'train': f'file://{train_dir}',
          'test': f'file://{test_dir}'}

estimator = TensorFlow(entry_point='train.py',
           model_dir=model_dir,
           train_instance_type=train_instance_type,
           train_instance_count=1,
           hyperparameters=hyperparameters,
           role=sagemaker.get_execution_role(),
           base_job_name='tf-eager-scriptmode-bostonhousing',
           framework_version='1.12.0',
           py_version='py3',
           script_mode=True)

estimator.fit(inputs)

results = predictor.predict(x_test[:10])['predictions'] 
```

The model is saved during training (to be checked).

3. Use the saved model anywhere, for instance:
	- using the Amazon SageMaker hosted endpoints functionality (save using TensorFlow SavedModel format in this case)
		```
		predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
		```
	- other way?
	- ...

## Amazon SageMaker local mode training

Convenient way to train locally a model for few epochs, from data contained locally (same location as notebook) or also on S3. Done to check that the code is working.

Pre-requisite: Docker Compose or NVIDIA-Docker-Compose (for GPU) installed on the notebook.

ex: `train_instance_type='local'`

## Amazon SageMaker hosted training

Used to perform the full training where the data is contained in S3. Can use significantly bigger resources with `train_instance_type` param.

ex: `train_instance_type='ml.c4.xlarge'`



