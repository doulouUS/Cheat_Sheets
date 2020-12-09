Full Stack Deep Learning

# Setting up a ML Project

## Lifecycle

Not linear, possible to iterate and come back to earlier step.

### Per project activities:

1. planning and project setup
	Goals and requirements

2. Data Collection and labeling

3. Training and debugging
	Implement a baseline
	Find SOTA and reproduce
	Improve iteratively on this

	In case of overfitting: might need to collect more data, better quality.

4. Deploying and testing
	Small prod test
	Write tests
	Move to full scale prod

	Possible to come back to 3. or 2. or even 1. if the chosen metrics is not the right one.

### Cross project infrastructure
	
	Team and hiring
	Tools and infra

### Other needed things

	understand state of the art in our domain
		Find one or two landmark results
		Go to who they cite, or who cited it
	promising areas of research, coming up

	Bring pb to upper management, convince them to use ML: 
		ML projects do not follow the same steps as a traditional engineering. You don't know what will work or not.
		Find places with quick wins to convince the management first.


## Prioritizing projects

### High impact ML pb

High impact and high feasibility

Mental models to identify high impact ML models:
	- where can you take advantage of cheap prediction?
	- where can you automate complicated manual processes?

	- less valid but still: anything that a human takes less than a second to realize, can be done by AI

### Assessing feasibility of ML project

First and foremost: is data available? At what cost? How much needed?

Second: accuracy requirement. Which confusion matrix are you ready to accept? Cost of wrong prediction?
It's exponentially costly to improve accuracy in terms of cost.

Third: difficulty of pb. Is it a well researched area already? If not may be more difficult. 
	How much compute was needed to get the results?

Still hard in supervised learning.
	- summarizing text
	- build 3d models
	- answering questions
	- predicting video
	- real-world reocgnition
	- symbolic reasoning


Hard problems:
	- where output is complex
	- reliability is required: failing safely out-of-distribution
	- generalization is required:  out of distribution data, small data.

Estimating cost per hour for ML projects
Evaluate cost of wrong predictions


## Archetypes

1. Improve an existing process

2. Augment a manual process

3. Automate a manual process

Tips in each of these archetypes:
(1.) Data Flywheels: collect more data, brings a better model, brings more users which brings more data and so on.
If you manage to create this loop, you can improve the impact of your system. Bring something good enough to production and iterate from there.

(2.) Create a product design that reduces the need for accuracy:
	ex: Facebook autosuggestion for tagging myself, asking if I want to be tagged
		Grammarly, with suggestions. Doesn't matter if it's wrong sometimes
		Netflix, saying why they recommend these things to you.

(3.) Add humans in the loop to gradually improve the fully autonomous process


## Metrics

Only one number is often optimized in ML models. Which one to choose? (ex. precision, recall, etc.)

2 ideas:
- Threshold the good metric ("it is good enough if the metric reaches this much"). Threshold can come from the domain knowledge, or can be set relative to an existing baseline.
- Optimize the bad metrics ("try to get the best value for it")

Process
1. Enumerate your requirements

2. What is the current performance of a basline

3. Which requirements is already good? Bad?
	Threshold the good one, and focus on optimizing the bad one
4. You can ignore some requirements at first
5. Revisit metrics to incorporate the ignored metrics or some needed refinement

Reduce bias in human in the loop system: find samples where the model is biased against, and go back to your data collection process to find more of these samples.


## Baselines

How do you know how well your model performs in an absolute manner?

Baseline gives you a lower bound on expected performance. Find them in:
- business/engineering requirements
- Published results (to be adapted though)
- Compare to rule-based 
- Use simple ML baselines: but prefer the next one, human baseline
- You can use human perf: from Turk < ensemble people < domaine experts < deep domain experts < mixture of experts

Use the best option you can afford, thinking that you may need to go back to collect more data. So asking an expert for one hour only may not be good.
Concentrate the time of your best labelers to hard tasks.

In tight deadlines, people would generally skip peaking baselines. This is not a good thing, having good baselines help you debugging your system.

It is a possible way to solve a pb with a big NN first and then prune it to adapt to your constraints (mobile).

# Infrastructure and Tooling

Reality of ML project
- aggregate, clean, store, label, version the data
- write and debug model code
- provision compute resources
- run experiments, store results
- test and deploy models
- monitor predictions and close the flywheel loop!

Different things  to consider:
1. Data:
	Cloud or local
	How do you store it
	data workflow
	versioning

2. Development and Training/Evaluation
	Cloud or local
	software engineering, frameworks, distributed training
	resource management, hyperparameter tuning

3. Deployment
	Cloud or local
	CI/Testing
	Web deployment
	Monitoring
	interchange format like ONNX

Some companies try to do all-in-one systems.

## Software engineering

programming language: python today

editors: vim, emacs, jupyter, VS Code, PyCharm
	VSCode would be his recommendation. 
		Open remotely projects
		Peek documentation
		Lint code as you type

If you  have code style it should be codified, ideally automatically applied

Use static analysis (method of debugging by automatically examining source code before running a program)
Combine this with dynamic analysis (unit testing, run after the code is run)

Use Python type hints

Use jupyter notebooks as first draft for a project, not so good though
	no version control
	nice to use as a documentation to illustrate with results how your code works
	hard for distributing task, or long task

Streamlit
	package a small applet to share your project
	use function decorators to have an applet coming
	smart caching, potentially 
	similar to databricks?

VS Code extension
	python

## Compute and GPUs

Development
	write code
	debug models
	looking at results
	you want to quickly compile models

Training/Evaluation
	model architecture/hyper param search

GPUs
	TPUs are the fastest current option
	RAM should  fit meaningful batches of your model: especially important for recurrent model
	Tensor Cores are good for convolutional/transformer models

	1080Ti is still ok to buy used (no computation mixed precision mode though but still a lot of memory)
	Turing and Volta are the current one to buy
	V100: fastest after TPU.

	Providers: AWS, GCP, small ones: Lambda labs, Paperspace

	timdettmers: PhD student blog to set up your GPU

	Makes sense to use GPUs for a lot of problems now: as long as you have a library 
	ex. rapid's AI library, pandas for GPU

## Resource management

	How to allocate resources to teams?

	SLURM workload manager: open source job scheduler for linux

	Docker (package the app) + Kubernetes (run many dockers on top of a cluster)

	Kubeflow
		spawn and manage jupyter notebooks
		allocate resources on top of a cluster of GPUs
		can manage multi-step ML workflows: allocate specific resources needed for each step
		similar to what kubernetes provide


## Frameworks and distributed training

Good for production: Caffe (2013), Tensorflow (2015)

Good for development: PyTorch (2017), built in python and run in python (you can debug your code by stopping it and see everything happening in the python code)

Intersection: Keras (2015)

Tensorflow is moving towards better development capabilities with eager execution in Tensorflow 2.0
PyTorch is getting better at production with TorchScript (from python, create an optimized graph that can run on many different things like mobile) in PyTorch 1.0.

FastAi is good to start with strong good models.

### Distributed training

Good for long iteration time or if lots of data.

Weights are tied on all GPUs but each GPU gets a different batch

Expect a 1.9 speed up for 2 GPU, 3.5 for 4 GPUs.

Model parallelism if model does not fit in a single GPU. Weights are partitioned in different GPUs.

Data Parallel on different GPU but a single Machine in Tensorflow: MirrorData..()
Data Parallel on different GPU but a single Machine in Pytorch: model = nn.DataParallel(model)

Look at Ray for DeepLearning but also for other use cases, if you want distributed computing (several machines each having several GPUs).
Horovod: Uber distributed training for Tensorflow, Keras, PyTorch, using MPI


## Experiment management

You can lose track of which code param and dataset used with which trained model, etc.
- Bad way: spreadsheet
- Tensorboard: good for single experiments, not good in the long term
- Losswise: everytime running a model there is a git commit, track something, then 
	comet.ml is an example
	weights & biases (used in class)
- MLFlow: opensource platform for ML machine learning, open source, good to self host and developped by DataBricks


## Hyperparameter tuning

Hyperopt for python: not only for deeplearning
Hyperas: wrapper around keras + hyperopt

Sigopt: startup Hyper parameter tuning SaaS.

Ray Tune: implement SOTA algorithm: hyperband

Weights and Biases: also does it.


## All-in-one solutions

SageMaker: 40% markup over corresponding EC2 instances

Netpune Machine Learning Lab

Domino Data Lab: recommended

# Data Management

## Sources

Only fields where you don't need a lot of data: RL, semi-RL, GANs
Nice to start on public data and deploy and then collect more data from users.

Data Flywheel: good way to enrich your data overtime.
	ex. Google photos asking you to confirm which face belongs to who in your photos.
		Good to have an initial model with great precision, but maybe low recall

Semi-supervised learning:
	NLP datasets: from the first 2 words predict the third.
	Try to obscure part of the data that becomes a label
	Image data augmentation: must do. Idea: inject world knowledge/signal not already present in the dataset
	Other data augmentation: idea, create new data that wouldn't fool a human but still constitutes new inputs for the model
		tabular:  delete some cells
		text: replace words with synonyms
		videos/sound: mask some frequencies, etc.

	synthetic data
		receipt: andrew Moffat> metabrite-receipt-tests, using blender to simulate crinkles on receipts
		driving cars, robotics: from simulations

	time series
		stretch or expand, compress 
		add uncorrelated noise to the data

	Technique to deal with difficult samples that are rare: 
		1. weights them more in your loss
		2. Train on the whole dataset - identify where it is wrong - recreate a dataset with a higher weight for where it gets things wrong. "Focal loss"


## Labeling

	- user interface
		usually comprises: bbox, segmentations, keypoints, etc.
		important to keep an eye on the annotation quality

	- source of labor
		can hire your own: secure, fast, less quality check needed. But expensive, slow to scale, admin overhead
		can crowdsource: cheap scalable, not secure, QC needed

	- service companies
		makes sense to outsource data labeling
		dedicate several days to selecting the best service company
			- Label yourself to get the full complexity
			- Choose based on results

			- FigureEight: being around for a while
			- Scale.ai: looks like an API call
			- Supervisely

		This service is pricy. Good to use your own labor but their software
			ex. Prodigy: has active learning

		Hiring part-time makes more sense than trying to make crowdsourcing work (QC important)

		Ensembling data labels:
			how to ensemble for subjective labels. ex. ranking
				see which one has consensus, which not
				predict the average rank instead of the rank itself

		Ex. Platform.ai to label

## Storage

### Filesystem

Can be networked NFS to share files
A file can be distributed (ex. HDFS): machine access this file not knowing where it is physically but it is part of a system like HDFS.

Careful with access patterns: if all your data is in a single disk, you cannot leverage parallelism.


### Object Storage

API over a file system:
GET, PUT, DELETE files.

You don't worry where the files are. The fundamental unit is an object. Versioning can be built in the system. Redundancy can be built-in by requiring a new object to be stored over multiple machines. Therefore can be parallel, but it is not fast.

Equivalent of S3: Ceph for on-prem


### Database

For structured data, it is fast and scalable and persistent.

Performing joint is not expensive cause it's like your data is sitting on your RAM, however in case of any disruption your data remains available on disk: the software make sure everything is logged and so everything is actually stored.

Not for binary data. The larger the DB the smaller the performance is, as things are in RAM.
Instead, store reference in your DB, referencing images in S3 (path to S3).

Use it for data that will be accessed frequently, not logs for instance.

Postgres: good choice most of the time, support unstructured JSON too. Under active development.

### Data Lake

Aggregation of data from multiple sources. Can be logs, db, data transformed at great expenses.

Schema on read: schema imposed when you read, i.e. when you actually need to come back to make use of the data. This is opposed to a database where you have a schema on write, as you add data.

Amazon Redshift - Snowflake: good solution for data lake.

### Where to put things?

binary data (sound, image, compressed texts): objects
metadata (labels, user activity): database
for features obtained from say logs, direct them to a data lake

for training time: copy needed data to filesystem (local or network): minimize distance between data and GPU.
-> Book: Designing Data-Intensive Applications.

It is reasonable to featurize and directly store this.

## Versioning

Levels

0: Do no versioning. Problem: your initial data helps produce a given model. In the future, you retrain this model but your data has changed. You should have the data associated with each model that has been trained on.

Version model+data!	not just model, cause if you modify the data and your model does not work anymore, you cannot go back and get your working model back.

1: everytime you train, take a snapshot of everything at training time. Not great, hacky

2: version data as a mix of assets and codes.
	Use git-lfs (large file storage): add this to your git. If I ever commit a JSON file to the repo, the hash of their content is stored in S3 in the back. Hash defines the version of the dataset. Add a timestamp on top.

	Can use lazydata too: lazyloading of data. Don't download until you need it.

3: Specialized solution, should be avoided unless expert knowledge.
	DVC (version data, version transformation applied to data)
	Pachyderm (version control data, language agnostic)
	Dolt from liquid data (company name...) good solution for versioning databases



	
## Processing (data workflow)

Different sources of data = different dependencies

Some tasks can only start when others finished.

Solution:
1. Makefile
	Limitations: recomputation depending on content, dependencies that are not file but databases and disparate programs, work spread over multiple machines

2. Airflow for data workflow (current winner)
	Create a DAG,
	A queue is created, job restarted if needed, UI

	=> But don't over engineer: unix command could save you, see "Command-line Tools can be 235x Faster than your Hadoop cluster". Pipes | happen in Parallel on different CPUs.


# Machine Learning Teams



# Training and Debugging

Try to fit a single batch asap!

## Overview

Causes of bad performance:
- Implementation bugs
	Ex. `glob` does not return files in a deterministic order
- Hyperparam choices
	Very subtle different choices of hyperparam can change widely results (ex. weights initialization)
- dataset construction
- data/model fit
	data is harder (ImageNet vs your dataset)

## Start Simple

- Simple Architecture: creating you a good benchmark already and then move to more complex architecture
	* Images: LeNet -> ResNet
	* Sequences: LSTM/temporal conv -> Attention model or WaveNet model
	* Other: FCN with one hidden layer -> Problem dependant

You can compose these architectures for a problem using a mix of these possible inputs (Images/Sequences/Other). Do not use pre-trained model initially.

- Use sensible hyperparams
	* Optimizer: Adam 3e-4
	* Activation: Relu (FC and Conv models), tanh (LSTMs)
	* No regularization, normalization initially (introduces bugs)

	Do not perform hyperparam search at this point (maybe learning rate a bit)

- Normalize inputs: substract mean and divide by variance
	* Crucial step
	* Images: scale to [0, 255]

- Simplify the problem at first: 
	* Reduce number of classifications
	* Reduce size of the training dataset, image size
	* Use a synthetic dataset

	=> When dealing with a difficult task, this shows if the lack of performance is coming from your implementation or the difficulty of the problem itself. Allows you to iterate faster, by training on your laptop first.


## Debugging

1. Get your model to run (not always easy)
	* Shape mismatch/Casting issue: go through each line with debugger
		* PyTorch: ipdb
		```python
		import ipdb; ipdb.set_trace()
		```
		* TF1.0: step throught the graph creation itself and look at shapes, then use a session
		```python
		sess = tf.Session()
		for i in range(num_epochs):
			import ipdb; ipdb.set_trace
			loss_, _ = sess.run([loss, train_op])
		```
		* Use tfdb: stops execution at each sess.run() and lets you inspect.
		```bash
		python -m tensorflow.python.debug.examples.debug mnist --debug
		```

		* Confusing tensor.shape, tf.shape(tensor), tensor.get_shape(), reshaping things to a shape of type Tensor
		* Flipped dimensions when using tf.reshape, take sum average or softmax over wrong dimension, forgot to flatten after conv layers, data stored as different dtype than loaded.

	* OOM: scale back memory intensive operations one-by-one (batch size, matrix sizes, etc.)
	* Other: google

2. Overfit a single batch of data
	Training loss should be going as close as we want to zero

	* Error going up:
		- Flipped sign of the loss function/gradient
	* Error explodes:
		- Numerical issue, high learning rate
	* Error oscillates:
		- decrease learning rate, inspect data and labels that may be corrupted (wrong preprocessing/shuffled, etc.)

	* Error plateaus: learning rate too low, gradients not flowing back through the whole model, too much regularization, incorrect input to loss function.

3. Compare to known results

	Find an official model implementation, run and compare both model line by line. 
	Be careful, many implementation on GitHub have bugs...

Most common bugs:
* Incorrect tensor shapes: can fail silently, with hidden broadcasting changing tensor names
* Pre-processing inputs: 
	ex. normalize data twice
	too much data augmentation

* Incorrect input to loss function
* Forget to set up train mode: batch norm then causes trouble
* Numerical instabilities: from exp or log or division operation.

General advice for implementation:
* Lightweight: less than 200 lines of codes
* Reuse components like Keras
* Data pipeline shouldn't be complicated yet: dataset should load to memory
* Batch norm bugs: if `train` flag not turned on, it forgets to update BN statistics


TODO
	Use VS Code
	Add open remote projects
	Git managing
	Lint code
	static analysis
	use  python typing
	Look at Jeremy Howard videos from fast.ai course for notebooks
	Use pytest
	Try steamlit
	Tensor TFlops vs 32bits
	flops?

	use git-lfs and lazydata
	reimplement papers:
		first a well implemented paper for checking
		then a more advanced to stand out



