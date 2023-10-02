#!/usr/bin/env bash
echo -e "Generating training data..."
python process_data.py --link-data data/train-0-images-boxable-with-rotation.csv --labels data/all-ann.csv --vocab data/vocab.csv > data/tmp.csv
echo -e "Generating test data..."
python process_data.py --link-data data/test-images-with-rotation.csv --labels data/all-ann.csv --vocab data/vocab.csv > data/test.csv

echo -e "Splitting into train/valid..."
sed -n '1,5001p' data/tmp.csv > data/valid.csv
sed -n '1p' data/tmp.csv > data/train.csv
sed -n '5002,$p' data/tmp.csv >> data/train.csv
rm data/tmp.csv
