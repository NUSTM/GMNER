# README

Here are code and dataset for our ACL2023 paper: [Grounded Multimodal Named Entity Recognition on Social Media](https://aclanthology.org/2023.acl-long.508.pdf)

## Updates

### 20230728: Twitter10000 v2.0

We have made some revisions to the Twitter10000 dataset.  In Twitter10000 v2.0, we made several detailed revisions to the BIO tagging and bounding box annotations, improving the alignment between the two to ensure a more accurate and consistent relationship.

## Dataset

Our dataset is built on two benchmark MNER datasets, i.e., Twitter-15 (Zhang
et al., 2018) and Twitter-17 (Yu et al., 2020).

- The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
- Step 1：Download each tweet's associated images via this link (<https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view>)
- Step 2:  Use [VinVL](https://github.com/pzzhang/VinVL) to identify all the candidate objects, and put them under the folder named "Twitter10000_VinVL". We have uploaded the features extracted by VinVL to [Google Drive](https://drive.google.com/drive/folders/1w7W4YYeIE6bK2lAfqRtuwxH-tNqAytiK?usp=sharing) and [Baidu Netdisk](https://pan.baidu.com/s/1QqjOlAAjCqAk_qL6ejeARw?pwd=TwVi) (code: TwVi).

## Requirement

- pytorch 1.7.1
- transformers 3.4.0
- fastnlp 0.6.0

## Usage

### Training for H-Index

```
sh train.sh
```

### Evaluation

```
sh test.sh
```

## Acknowledgements

- Using the dataset means you have read and accepted the copyrights set by Twitter and original dataset providers.
- Some codes are based on the codes of  [BARTNER](https://github.com/yhcc/BARTNER), thanks a lot!
