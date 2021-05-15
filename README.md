# SoundGAN - Seeing Sound: Estimating Image from Audio 

Ongoing (strated at November 2020) Final Project at the Technion - Israel Institute of Technology.

Project supervisor: Tamar Rott Shaham, Idan Kligvasser.

Project Goal - Create an estimated image of a given sound source.

## How To Use This Code

  * Install requirements.txt with pip. 
  * Download sub-URMP dataset to destination folder - for training.
  * Download desired test data to test folder.
  * Run run_SoundGan.py with desired mode (i.e generate, train, loss, audio classification, bigGAN and more), check utils.py for more details on file arguments.

## Architecture

![Architecture](imgs/Architecture.png?raw=true "Architecture")

## Current results

![Results](imgs/Results.png?raw=true "Results")

## Misc Notes
Git does not include databases, checkpoint's and some other required files to run, only code.

Final Git version will be done, once project is finalized.

## Cites

[1] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).

[2] L. Chen, S. Srivastava, Z. Duan and C. Xu. Deep Cross-Modal Audio-Visual Generation. In Proc. of ACM International Conference on Multimedia Thematic Workshops, 2017.

[3] B. Li, X. Liu, K. Dinesh, Z. Duan and G. Sharma. Creating A Musical Performance Dataset for Multimodal Music Analysis: Challenges, Insights, and Applications. arXiv:1612.08727, 2016. (The original URMP dataset paper.) 

[4] Andrew Brock, Jeff Donahue, Karen Simonyan. [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096).
