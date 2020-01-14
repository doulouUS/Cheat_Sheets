

# Dockerize

## Build an image

* You will need to create a `Dockerfile` containing the instructions executed by the `docker build` command. It is like an installer script. See `Dockerfile`.

* Build the container image. `-t` sets the name and tag for the new container image. `.` indicates the base directory (context) where the container is to be built, it is the same as where the `Dockerfile` is written.
```
docker build -t dashboard_viewer:latest .
```

* Any changes to the application? Rebuild the container image with the previous command.
* List available images:
```
docker image ls
```

Docker is a client-server application with Docker server called *daemon*

## Start a container 

Once you have the image, you can launch the app with
```
docker run ...
```

There are many arguments:
* `--name` name for the new container
* `-d` run the container in the background
* `-p` maps container ports to host ports. Ex: 8000:5000, port of the host computer:port inside the container
* `--rm` delete the container once terminated. This is not mandatory but recommended
* `-e` run-time environment variables. Ex:`-e SECRET_KEY=secret-key`
* `--link` make another container accessible to the current one. Ex: mysql:dbserver, name of the container to link:hostname used to refer to the linked one
* last argument is the name of the container image and the tag separated by `:`


`docker run` outputs an ID identifying the container running

`docker ps` shows which containers are running.

## Stop the container

```
docker stop ID_of_the_container
```

## Saving data outside of the application container

Create additional containers, e.g. for a MySQL database. The `docker run` will then add more options to connect these containers.

Start a MySQL server:
```
docker run --name mysql -d -e MYSQL_RANDOM_ROOT_PASSWORD=yes \
    -e MYSQL_DATABASE=dashboard_viewer -e MYSQL_USER=dashboard_viewer \
    -e MYSQL_PASSWORD=<database-password> \
    mysql/mysql-server:5.7
```

The application side needs a MySQL client, just need to add this to the `Dockerfile`:
```
...
RUN venv/bin/pip install gunicorn pymysql
...
```

Final command linking the `dashhboard_viewer` application to the previously defined MySQL 

```
docker run --name dasboard_viewer -d -p 8000:5000 --rm -e SECRET_KEY=my-secret-key \
    -e MAIL_SERVER=smtp.googlemail.com -e MAIL_PORT=587 -e MAIL_USE_TLS=true \
    -e MAIL_USERNAME=<your-gmail-username> -e MAIL_PASSWORD=<your-gmail-password> \
    --link mysql:dbserver \
    -e DATABASE_URL=mysql+pymysql://dashboard_viewer:<database-password>@dbserver/dashboard_viewer \
    dashhboard_viewer:latest
 ```

#  `Dockerhub`

## Retrieve an image from [`Dockerhub`](https://hub.docker.com/)

If running an image that is not stored locally, it is downloaded from Dockerhub
```
docker run python:3.7-alpine
```

To just pull the image without running it:
```
docker pull image_name
```

## Publish an image to Dockerhub

Similarly to Github, login to [Dockerhub](https://hub.docker.com/) and *Create Repository* and choose a name.

Log into Dockerhub using the CLI
```
docker login --username=yourhubusername --email=youremail@company.com
```

With your docker image ID obtained by running `docker images`, tag your container image:
```
docker tag container_image_ID yourhubusername/your_container_name:your_container_tag
```

Finally push the image to Dockerhub
```
docker push yourhubusername/your_container_name
```


# Random

Explore instance image content in an interactive shell
```
docker run -it image_name bash
```

Clean up unused or dangling images, containers, volumes and network
```
docker system prune  // add -a to remove stopped containers and unused images as well
```

# Removing

Comprehensive guide [here](https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes)

## Removing containers

Removing all exited containers
```
docker rm $(docker ps -a -f status=exited -q)
```

Remove all containers
```
docker rm $(docker ps -a -q)  // add -f to force delete (when dependencies and taggs)
```

## Removing images

Remove image using its ID
```
docker image rmi ImageID
```

Remove all images
```
docker rmi $(docker images -a -q)
```


# `.dockerignore` syntax

```
pattern:
{ term }
term:
'*' matches any sequence of non-Separator characters
'?' matches any single non-Separator character
'[' [ '^' ] { character-range } ']'
character class (must be non-empty)
c matches character c (c != '*', '?', '\\', '[')
'\\' c matches character c
 
character-range:
c matches character c (c != '\\', '-', ']')
'\\' c matches character c
lo '-' hi matches character c for lo &lt;= c &lt;= hi
 
additions:
'**' matches any number of directories (including zero)
'!' lines starting with ! (exclamation mark) can be used to make exceptions to exclusions
'#' lines starting with this character are ignored: use it for comments
```


# Automated image testing and build

Set up `docker-compose.test.yml` file at the same location as the `Dockerfile`. 
...
