# Week 1:

This directory includes the exercises for the 1st week.

The exercises includes:

* Exercise1: Testing and learning how to use gym environments
* Exercise2: Implement Evolutionary Hill-Climber with simple Neural Network and train it.

To run the exercises, a docker file has been created from the team of the course as ready environment.

```bash

# Download the container
docker pull vkurenkov/cognitive-robotics

# Run container
docker run -it \
  -p 6080:6080 \
  -p 8888:8888 \
  --mount source=cognitive-robotics-opt-volume,target=/opt \
  vkurenkov/cognitive-robotics

```

Then open http://127.0.0.1:6080/ from browser to access the novnc server, that is running by the docker image.

To open a terminal in that interface, click right-click -> Applications -> Shells -> Bash

To develop the codes using VScode, go to extension manager and install remote-containers, then press the key on the most down left, then attach to running container and you will be in the container directories,

