
# Setting up AWS CLI

AWS Access Key and Secret Access Key are my credentials. Associated with an IAM user or role

It is recommended to create a new administrator IAM user instead of using the root one. Save the AWS Access and Secret Key.

Next, configure a default user (you can create more users with their own name)
```
aws configure
```

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

Structure correctly your JSON by generating a JSON skeleton that you can then filled correctly:
```
$ aws2 ec2 describe-instances --generate-cli-skeleton  // output:
{
    "Filters": [
        {
            "Name": "",
            "Values": [
                ""
            ]
        }
    ],
    "InstanceIds": [
        ""
    ],
    "DryRun": true,
    "MaxResults": 0,
    "NextToken": ""
}
```


# Launching an EC2 instance

* Find the AMI ID among the Amazon ones:
```
aws ec2 describe-images --owners self amazon`
```

* Find the settings of your existing instances to reuse them (ex. reuse the default subnet id etc.)
```
aws2 ec2 describe-instances  // --filters "Name=instance-type, Values=t2.micro,m1.medium" // for further filtering among your instances
``` 

# Launching a Docker app on an EC2 instance

1. Launch an EC2 instance (see above) and wait for it to be available

2. Install Docker [manually](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

Add docker to the current user (here `ubuntu`) to avoid using `sudo` all the time
```
sudo usermod -a -G docker ubuntu
```

3. Save the local image you need to a tar file 
```
docker save -o <path for generated tar file> <image name>
```

4. Move it to your new host
```
scp -i "key.perm" path/to/local/image user@remote_host:remote_dir
```

5. Load the image
```
docker load -i image.tar
```

Disadvantages: automation, scaling, not integrated in a robust infra (load balancer etc.) => use ECS


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
