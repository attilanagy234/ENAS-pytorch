#!/usr/bin/env bash

set -e

TIMESTAMP=$(date "+%Y%m%d%H%M%S")

echo "######################################"
echo "BUILDING IMAGE WITH TAG $TIMESTAMP"
echo "######################################"

docker build -t enas-tdk .

docker tag enas-tdk:latest 855453161598.dkr.ecr.eu-west-1.amazonaws.com/enas-tdk:latest
docker tag enas-tdk:latest 855453161598.dkr.ecr.eu-west-1.amazonaws.com/enas-tdk:$TIMESTAMP

docker push 855453161598.dkr.ecr.eu-west-1.amazonaws.com/enas-tdk:latest
docker push 855453161598.dkr.ecr.eu-west-1.amazonaws.com/enas-tdk:$TIMESTAMP