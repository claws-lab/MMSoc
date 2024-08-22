# MMSoc: Multimodal Social Media Analysis

## Project Overview

MMSoc is a large-scale benchmark for analyzing the performances of multimodal LLMs (MLLMs) in social media analysis.

* Paper: [MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms](https://arxiv.org/abs/2402.14154)

* [🤗HuggingFace Benchmark](hhttps://huggingface.co/collections/Ahren09/mmsoc-benchmark-66b0179b6581beb2ca80d740) 

```bibtex
@inproceedings{jin2024mm,
  title={MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms},
  author={Jin, Yiqiao and Choi, Minje and Verma, Gaurav and Wang, Jindong and Kumar, Srijan},
  booktitle={ACL},
  year={2024}
}
```


The [🤗 MMSoc benchmark](hhttps://huggingface.co/collections/Ahren09/mmsoc-benchmark-66b0179b6581beb2ca80d740) contains the following datasets

### Memotion [[🤗Link](https://huggingface.co/datasets/Ahren09/MMSoc_Memotion)]



* 12,143 memes, annotated by AMT with labels that categorize the memes according to their:
  * sentiment (positive, negative, neutral)
  * types of emotion they convey (sarcastic, funny, offensive, motivational)
  * intensity of the expressed emotion.
* **Modality**: images, embedded text
* **Tasks**: OCR, humor detection, sarcasm detection, offensive detection, motivation analysis, sentiment analysis


### Hateful Memes [[🤗 Link](https://huggingface.co/datasets/Ahren09/MMSoc_HatefulMemes)]

* 12,840 memes with meme-like visuals abd text laid over them. 
* **Modality**: images, embedded text
* **Tasks**: hate speech detection


### YouTube2M [[🤗 Link](https://huggingface.co/datasets/Ahren09/MMSoc_YouTube2M)]

* 2 million YouTube videos shared on [Reddit](https://www.reddit.com/)
* 62 unique tags
* 1,389,219 videos bearing the top 5 tags (70.7% of the dataset)
* **Modalities**: text, image 
* **Tasks**:
  * **tagging**: predicting appropriate “topic categories” for YouTube videos 
  * **text generation**: Generate the titles / descriptions of the videos

  * For ease of testing, we have also released a smaller sample of the dataset, [🤗 YouTube2000](https://huggingface.co/datasets/Ahren09/MMSoc_YouTube2000), with 2000 samples (1600 train, 200 validation, 200 test).

### FakeNewsNet

* We consider two datasets under the misinformation  detection theme:

  * [🤗 PolitiFact](https://huggingface.co/datasets/Ahren09/MMSoc_PolitiFact)
  * [🤗 GossipCop](https://huggingface.co/datasets/Ahren09/MMSoc_GossipCop)

* **Modalities**: news content (text), online posts (text), images, user metadata

The datasets were originally curated by [Shu et al](https://arxiv.org/abs/1809.01286) ([GitHub](https://github.com/KaiDMML/FakeNewsNet/)).

* **Tasks**: Misinformation detection


## Project Structure

* `mmsoc/`: main package directory for the MMSoc project.
* `models/`: Sample code for using the dataset
* `blip.py`**: This file includes the implementation of the BLIP2 (Bidirectional Language Image Pretraining) and InstructBLIP models, which is used for tasks that require joint image and text understanding.

## Installation

To get started with MMSoc, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd MMSoc
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   conda create -n mmsoc python=3.11
   conda activate mmsoc
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).
