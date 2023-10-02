#!/usr/bin/env bash
# TODO: Automate downloading of images

# Download images
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz downloaded
aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz downloaded
aws s3 --no-sign-request cp s3://open-images-dataset/tar/test.tar.gz downloaded

# NOTE: we copy pasted these links from: https://storage.googleapis.com/openimages/web/download_v6.html
# Section Subset with Image-Level Labels (19,958 classes)
# Choice: Human-verified labels  (We do not want the machine generated ones)
mkdir -p labels
wget -O labels/train.csv https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv
wget -O labels/valid.csv https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels.csv
wget -O labels/test.csv https://storage.googleapis.com/openimages/v5/test-annotations-human-imagelabels.csv

# Download mapping from label hashes to their description
wget -O trainable.csv https://storage.googleapis.com/openimages/v6/oidv6-classes-trainable.txt
wget -O classnames.csv https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv

mkdir -p downloaded
ls --color=never downloaded/train_1 | cut -f1 -d"." > downloaded/train.csv
ls --color=never downloaded/validation | cut -f1 -d"." > downloaded/valid.csv
ls --color=never downloaded/test | cut -f1 -d"." > downloaded/test.csv
# NOTE: We did not use https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/OpenImages.md
# Which was from the paper, since for the training data
# it includes machine generated labels
mkdir -p data

python build_vocab.py --labels labels --downloaded downloaded --trainable trainable.csv --classnames classnames.csv
head -n 10001 test.csv > test-10k.csv
