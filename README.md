<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ntropy-network/enrichment_models">
    <img src="https://developers.ntropy.com/img/ntropy-logo.svg" alt="Logo" width="80" height="80">
  </a>

  <h2 align="center">Financial transaction models benchmark</h2>

  <h3 align="center">This repository benchmark Ntropy API against different Large Language Models (OpenAI ChatGPT and LLAMA finetuned models). It also contains an easy to use wrapper to enable using the LLMs to perform transaction enrichment. Llama adapters were open sourced and are available on huggingface hub. </h3>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#benchmark">Benchmark</a></li>
    <li><a href="#usage">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#ressources">Ressources</a></li>
  </ol>
</details>



<!-- BENCHMARK -->
## Benchmark

We benchmarked Ntropy's API and a set of LLMs in the task of extracting the following fields: label, merchant and website.

Ntropy's API is compared against:
- OpenAI's LLM's (GPT-4) using a straightforward prompt.
- Llama finetuned models (7B & 13B params) on consumer transactions data with LORA adapters.

The dataset used can be found here: [`/datasets/100_labeled_consumer_transactions.csv`](https://github.com/ntropy-network/enrichment_models/blob/main/datasets/100_labeled_consumer_transactions.csv).
All predictions can be found here: [`/datasets/benchmark_predictions.csv`](https://github.com/ntropy-network/enrichment_models/blob/main/datasets/benchmark_predictions.csv).
It consists of a random subset of 100 anonymized consumer transactions.
The full label list can be found [here](https://api.ntropy.com/v2/labels/hierarchy/consumer).


|                           | GPT 4 | LLAMA finetuned 7B | LLAMA finetuned 13B | Ntropy API |
|---------------------------|-------|--------------------|---------------------|------------|
| Labeler Accuracy          | 0.71  | 0.72               | 0.78                | **0.86**       |
| Labeler F1 Score          | 0.64  | 0.56               | 0.65                | **0.73**       |
| Labeler Label similarity * | 0.85  | 0.82               | 0.87                | **0.91**       |
| Labeler latency (s/tx)    | 1.47  | 0.27               | 0.34                | **0.01**       |
|                           |       |                    |                     |              |
| Merchant Accuracy         | 0.66  |         /           |          /           | **0.87**       |
| Website Accuracy          | 0.69  |         /           |          /           | **0.87**       |
| Normalizer latency (s/tx) | 4.45  |         /           |          /           | **0.01**       |



_*: Label similarity is an approximate metric that uses embeddings distance to give a smoother metric than the accuracy (ex: 2 similar labels will have a score close to 1 while 2 very different semantically will have a score close to 0).
You can see more details in `tests/integration/test_openai::test_label_similarity_score` ._

Among the models evaluated, Ntropy demonstrates the best metrics in terms of accuracy and latency. This superiority can be attributed to several factors, including its access to web search engines and internal merchant databases. Moreover, Ntropy's internal models have been fine-tuned specifically for financial tasks, contributing to their effectiveness to get accurate labels.

We noticed that when a Llama model is fine-tuned on consumer transactions, even without having access to external information about merchants, it achieves a higher accuracy compared to GPT-4 (by 7 points). This suggests that LLM's models possess a considerable amount of knowledge about companies, even though measuring this knowledge directly can be challenging. Additionally, retrieving cleaned company names and websites appears to be more difficult for these models.

Based on this dataset, it is 'funny' to note that GPT-4 has the ability to generate websites that appear to be correct at first glance but, in reality, do not exist. For instance:
- KwikCash => http://www.kwikcash.com/ (instead of https://www.kwikcashonline.com/)
- Clark's Pump-N-Shop => https://pumpnshop.com/ (instead of https://www.myclarkspns.com/)
- ...

Note: LLAMA models were benchmarked on a single A100 GPU.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- INSTALLATION -->
## Installation

_This project uses python>=3.10_

Python package that can be installed either using poetry or pip:

- Poetry:

```
poetry install
poetry shell
```

- Pip:

```
pip install .
```

Depending on which model you want to run, you need at least one of the following (or all for running the full benchmark):

### Ntropy API Key

For using the Ntropy API, you need an API KEY:

- Go to https://dashboard.ntropy.com/
- Create an account (you can use your google account or any mail)
- On the left-side menu, you can click on "API Keys" and then click on "Create API Key"
- Copy the API KEY and paste it here: [`enrichment_models/__init__.py`](https://github.com/ntropy-network/enrichment_models/blob/main/enrichment_models/__init__.py)

Note: You will get a limit of 10 000 transactions with a free account! If you need more, please contact us.

### OpenAI API Key

For using the OpenAI models, you'll need an API KEY:

- Go to https://platform.openai.com/
- Create an account
- On the dropdown menu click on "View API keys"
- Then, "Create new secret key"
- Copy the API KEY and paste it here: [`enrichment_models/__init__.py`](https://github.com/ntropy-network/enrichment_models/blob/main/enrichment_models/__init__.py)


### LLAMA requirements

The LLAMA adapters are open sourced and can be used from the HuggingFace Hub.
The models have 2 variants (7B params & 13B params, 16bits) and can be found at the following urls:

-  https://huggingface.co/ntropydev/ntropy-labeler-llama-lora-7b
- https://huggingface.co/ntropydev/ntropy-labeler-llama-lora-13b

Note: Minimum 32gb of RAM are needed to run LLAMA models (better if you have access to some GPU's with enough VRAM)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### Benchmark

If you want to run the full benchmark, after setting up API KEY's in [`enrichment_models/__init__.py`](https://github.com/ntropy-network/enrichment_models/blob/main/enrichment_models/__init__.py), you can just run:

```
make benchmark
```

Or

```
python scripts/full_benchmark.py
```

This will print results on the terminal as well as dumping metrics and predictions in the `datasets/` folder.

### Model Integration

If you want to integrate one of these models, you can just take examples on the notebooks, in the `notebooks/` folder.

Also, if you want to integrate Ntropy's API, you might want to have a look at the [documentation](https://developers.ntropy.com/)

There is one notebook per model (ntropy, openai and llama).


<!-- CONTRIBUTING -->
## Contributing

We welcome and appreciate any Pull Request that suggests enhancements or introduces new models, APIs, and so on to add to the benchmark table.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`](https://github.com/ntropy-network/enrichment_models/blob/main/LICENSE) for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

[support@ntropy.com](support@ntropy.com)


<!-- ACKNOWLEDGMENTS -->
## Ressources

Main project dependencies:

* [Alpaca LORA](https://github.com/tloen/alpaca-lora)
* [OpenAI Python](https://github.com/openai/openai-python)
* [Ntropy SDK](https://github.com/ntropy-network/ntropy-sdk)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ntropy-network/enrichment_models.svg?style=for-the-badge
[contributors-url]: https://github.com/ntropy-network/enrichment_models/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ntropy-network/enrichment_models.svg?style=for-the-badge
[forks-url]: https://github.com/ntropy-network/enrichment_models/network/members
[stars-shield]: https://img.shields.io/github/stars/ntropy-network/enrichment_models.svg?style=for-the-badge
[stars-url]: https://github.com/ntropy-network/enrichment_models/stargazers
[issues-shield]: https://img.shields.io/github/issues/ntropy-network/enrichment_models.svg?style=for-the-badge
[issues-url]: https://github.com/ntropy-network/enrichment_models/issues
[license-shield]: https://img.shields.io/github/license/ntropy-network/enrichment_models.svg?style=for-the-badge
[license-url]: https://github.com/ntropy-network/enrichment_models/blob/main/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/ntropy/
