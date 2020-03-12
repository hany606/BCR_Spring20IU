# BCR_Spring20IU
This repository is for the assignments and the labs of Behavioral &amp; Cognitive Robotics Technical Elective Course in Innopolis University for Spring 2020 as a free listener.

This repository is alongside exercises in the repository that has been used from the instructor of the course (Prof. Stefano Nolfi) in that [link](https://github.com/snolfi/evorobotpy)

## Notes to run the codes:

* To run the exercises/codes, a docker file has been created from the team of the course as ready environment.

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

* Then open http://127.0.0.1:6080/ from browser to access the novnc server, that is running by the docker image.

* To open a terminal in that interface, click right-click -> Applications -> Shells -> Bash

* To develop the codes using VScode, go to extension manager and install remote-containers, then press the key on the most down left, then attach to running container and you will be in the container directories,

* To add GitHub credientials especially if two factor authontication is enabled, check [here](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line)
