# General Command structure

```
aws2 top_level_command subcommand OPTIONS_or_PARAM
```

* `aws2` base call
* `top_level_command` AWS service usually
* `sub_command` operation to perform

To get help on how to format parameter for a specific subcommand
```
aws ec2 describe-spot-price-history help
``` 

If complex parameters or options, structure them using a JSON file instead:
```
aws ec2 describe-instances --filters '{"Name": "instance-type", "Values": ["t2.micro", "m1.medium"]}'
```

If multiple filters, encapsulate them between brackets. You can save this JSON parameters in a file and load it using this particular syntax:
```
aws ec2 describe-instances --filters file://*path_to_json_file*
```


# Launching a Docker app on ECS

Create cluster (of EC2 instances). Need to figure out options here!!
```
aws ecs create-cluster — cluster-name dashboard_visualization
```

## Steps (ECS+Load Balancer)

Through [GUI](https://aws.amazon.com/getting-started/tutorials/deploy-docker-containers/)

1. Launch cluster 
	* Possible to store images on Amazon Elastic Container Registry (ECR)

2. Task Definition
	* which Docker image to use
	* how many containers
	* resource allocation for each container

3. Configure my Amazon ECS *service*
	* configure service options
	* elastic load balancing 
	* link with an IAM role

4. Configure the cluster
	* cluster name, instance type, number of instances, key pair
	* set up security group
	* IAM role

5. Launch and test

6. Delete resources
