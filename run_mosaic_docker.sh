#!/bin/bash

DOCKER_NAME="tomp/mosaic-gradio"
MOSAIC_PORT=7860

SHORT_HOST=$(hostname --short)
LONG_HOST=$(hostname --long)

die () { echo "FATAL: $*"; exit 1; }

usage () {
    [[ -n $* ]] && echo "ERROR: $*"
   cat <<END
Usage: $0 [--port <port>]
Options:
    -p|--port ..... set port used to access the webapp (default: 7860)
    -h|--help ..... print this summary

This starts up a mosaic-docker container to provide a local instance
of the Mosaic web app.
END
    [[ -n $* ]] && exit 1
    exit 0
}

while [[ $# -gt 0 ]]; do
    ARG="$1"; shift
    case "$ARG" in
        -p|--port) MOSAIC_PORT="${1?missing port number}"; shift;;
        -h|--help) usage;;
               -*) usage "Unrecognized option '$ARG'";;
                *) usage "Unrecognized argument '$ARG'";;
    esac
done

[[ -n $HF_TOKEN ]] || die "HF_TOKEN is undefined"

echo "Access the Mosaic app at"
echo "    http://${SHORT_HOST}:${MOSAIC_PORT}/"
echo "or"
echo "    http://${LONG_HOST}:${MOSAIC_PORT}/"
echo
echo "You may need to wait a minute or so for the models to download before the site appears."
echo

# Get current user ID and group ID to ensure files have correct ownership
USER_ID=$(id -u)
GROUP_ID=$(id -g)

docker run -it \
    --gpus=all --runtime=nvidia \
    --user "${USER_ID}:${GROUP_ID}" \
    --env HF_TOKEN=${HF_TOKEN} \
    --shm-size=500m \
    -p ${MOSAIC_PORT}:7860 \
    docker.io/${DOCKER_NAME}

