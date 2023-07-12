# README

Here are code and dataset for our ACL2023 paper: [Grounded Multimodal Named Entity Recognition on Social Media](https://aclanthology.org/2023.acl-long.508.pdf)

## Dataset

Our dataset is built on two benchmark MNER datasets, i.e., Twitter-15 (Zhang
et al., 2018) and Twitter-17 (Yu et al., 2020).

- The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
- Step 1：Download each tweet's associated images via this link (<https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view>)
- Step 2:  Use [VinVL](https://github.com/pzzhang/VinVL) to identify all the candidate objects, and put them under the folder named "Twitter10000_VinVL"

## Requirement

- pytorch 1.7.1
- transformers 3.4.0
- fastnlp 0.6.0

## Usage

### Training for H-Index

```
sh rain.sh
```

### Evaluation

```
sh test.sh
```

## Acknowledgements

- Using the dataset means you have read and accepted the copyrights set by Twitter and original dataset providers.
- Some codes are based on the codes of  [BARTNER](https://github.com/yhcc/BARTNER), thanks a lot!