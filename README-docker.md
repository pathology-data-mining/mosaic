# Mosaic: Docker deployment

The Mosaic app been packaged as a Docker image, which may be easier to use than
installing the app in a Python environment.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

### System requirements

Supported systems:

- Linux (x86) with GPU (NVIDIA CUDA)

### Pre-requisites

You will need to have Docker or Podman installed on your system, and at least 8G of
storage space for the docker image.

You will need to have the NVidia Container Toolkit installed on the machine where
you want to run the Mosaic app.  For instructions, see
[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Installation

1. Pull the image into your local Docker repository

```bash
docker pull docker.io/tomp/mosaic-gradio
```

2. Set the HF_TOKEN to your HuggingFace access token

The models for Mosaic are not yet public.  To access the models, you need to be
a member of the [Pathology Data Mining Group](https://huggingface.co/PDM-Group)
organization on HuggingFace.

To download the models, you need to set the HF_TOKEN environment variable to your
HuggingFace access token.

If you don not already have one, create an access token by logging in to your account
of HuggingFace, and clicking on the user icon at the top right corner of the site and
selecting "Access Tokens". When creating the token, select all read options for your
private space and the PDM-Group space.

```bash
export HF_TOKEN="TOKEN-FROM-HUGGINGFACE"
```

## Usage

### Web Application

1. Start up the web app using the command

```bash
docker run -it \
    --gpus=all --runtime=nvidia \
    --env HF_TOKEN=${HF_TOKEN} \
    --shm-size=500m \
    -p 7860:7860 \
    tomp/mosaic-gradio
```

2. Access the webapp at the URL [http://localhost:7860/](http://localhost:7860)


*. You can also start up the docker container using the `run_mosaic_docker.sh` script
in this repo.  That executes the `docker run` command (shown above) for you, and lets 
you specify the port you want to use to access the app (if 7860 is not available).

To run it, you would just execute

```bash
./run_mosaic_docker.sh

or

./run_mosaic_docker.sh --port 7863
```

### Command Line Interface (CLI)

If you need to run Mosaic through Docker, use the provided `mosaic-docker` wrapper script.
It automatically handles volume mounting and passes all arguments to the containerized
Mosaic CLI.

If Docker is not required, prefer the native `mosaic` command documented in [README.md](README.md).

#### Basic Usage

```bash
# Show help
./mosaic-docker --help

# Process a single slide
./mosaic-docker --slide-path /path/to/slide.svs \
                --output-dir /path/to/output \
                --site-type Primary \
                --cancer-subtype Unknown \
                --segmentation-config Resection

# Process multiple slides from a CSV file
./mosaic-docker --slide-csv /path/to/slides.csv \
                --output-dir /path/to/output

# Process a breast cancer slide with IHC subtype
./mosaic-docker --slide-path /path/to/breast_slide.svs \
                --output-dir /path/to/output \
                --site-type Primary \
                --cancer-subtype BRCA \
                --ihc-subtype "HR+/HER2-"
```

#### How it works

The `mosaic-docker` wrapper script:
- Automatically mounts input slide directories and output directories into the container
- Passes through all Mosaic CLI arguments
- Handles the HF_TOKEN environment variable
- Detects and uses GPU support if available (falls back to CPU if not)

**Note**: When using `--slide-csv`, the script mounts the directory containing the CSV file.
For slides referenced in the CSV, they should be in the same directory as the CSV file or in
subdirectories relative to it. If slides are in different locations, you may need to modify
the CSV to use relative paths or run the docker command directly with additional volume mounts.

#### Requirements for CLI usage

- Docker installed and running
- HF_TOKEN environment variable set (same as web app)
- NVIDIA Docker runtime for GPU support (optional, will run on CPU if not available)

### Notes

- After you start up the application, it will download the necessary models from
HuggingFace. This may take some time (up to a few minutes) depending on your
internet connection.
