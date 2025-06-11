# artificial-authorship-verification
This repository contains the code associated with my Master thesis about authorship verification of artificial-generated texts and human-authored texts.
_Section with an asterisk * are generated with the help of ChatGPT._

## Contents
WIP: What did I do in this thesis?

Feel free to explore my written thesis [here as soon as public](https://github.com/KlaraGtknst/master-thesis).

## 🚀 Getting Started*

The project uses [**Poetry**](https://python-poetry.org/) for dependency management and packaging. 
[**Poetry**](https://python-poetry.org/) is a modern Python tool for **dependency management** and **packaging**. 
It replaces tools like `pip`, `virtualenv`, and `setuptools` with a single, streamlined workflow.

#### 🔧 Key Benefits*

- Manages dependencies via `pyproject.toml`
- Automatically creates isolated virtual environments
- Simplifies package building and publishing
- Provides a clean CLI for common tasks (`install`, `add`, `build`, etc.)


### ✅ Requirements*

- **Python ≥3.10 and <3.14**
- **Poetry ≥1.3**

Install poetry using the following command for MacOS:

```bash
brew install poetry
```

### ⚙️ Set up the project*
1. Clone the repository
2. Navigate to the project directory: `cd artificial-authorship-verification`
3. Install the dependencies using Poetry:
   ```bash
   poetry install
   ```

### 🛠️ Common Poetry Commands*
- Add a new dependency:
  ```bash
  poetry add <package-name>
  ```
- Update dependencies:
  ```bash
  poetry update
  ```


# 🗂️ Datasets

Before you can run the code, you need to download the datasets and place them in the `data/datasets` directory.
The datasets need to be converted to the Hugging Face dataset format.
After downloading the datasets defined in the following, and adding them as described below, you can use the provided script `genai_detection/dataset_util` to convert them.
Note that you have to comment dataset methods you do not need at the bottom of the script, and run the script to convert the datasets.

## 📁 Current datasets
- **Cross-Discourse PAN23**: [PAN23 dataset](https://pan.webis.de/clef23/pan23-web/author-identification.html#data) (Request access at Webis group)
  - Request access at [FoLD webpage](https://fold.aston.ac.uk/handle/123456789/17)
  - Cross-Discourse Type AV (e.g. essay vs. email) PAN@CLEF2023.
  - Contains pairs of texts with authorship information.
  - The dataset is in JSONL format.
- **Koppel et al. (2014)**: [Blog Authorship Corpus](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download)
  - Contains blog posts with authorship information.
  - The dataset is in CSV format.
- **Koppel et al. (2014) Student Essays**: WIP dataset (not available currently)
  - Contains student essays with authorship information.
- **Fanfiction (PAN20)**: [PAN20 dataset](https://zenodo.org/records/5106099) 
  - Contains fanfiction texts with authorship information.
  - The dataset is JSONL format.
- **Gutenberg**: [Gutenberg dataset](https://www.gutenberg.org/)
  - Contains texts from the Gutenberg project with authorship information.
  - The dataset is in TXT format.

## 📁 Add a new dataset
To add a new dataset, create new directories in the `data/datasets/<your-dataset-name>` directory with the following structure:

```
<your-dataset-name>/
├── <train>/
│   ├── pairs.jsonl
│   └── truth.jsonl
└── <test>/
    ├── pairs.jsonl
    └── truth.jsonl
```

Since we work with Hugging Face datasets, the `pairs.jsonl` file should contain a JSON Lines file with the following structure:

```json
{"id": "1", "pair": ["This is the first text.","This is the second text."], "same": "True", "authors": ["Author1", "Author1"]}
{"id": "2", "pair": ["This is the first text.","This is the second text."], "same": "False", "authors": ["Author3", "Author4"]}
```
To convert a dataset to a Hugging Face dataset, you can use the `genai_detection/dataset_util.py` script. 
Create a class that inherits from `BaseDatasetLoader` or `Pan23DatasetLoader` and implements the missing methods (like `Pan20DatasetLoader`).
Create a `run_<your-dataset>` function.
Run the `run_<your-dataset>` function from main at the bottom of the script.
This script will read the `pairs.jsonl` and `truth.jsonl` files and convert them into a Hugging Face dataset format.


# 📦 Enroot Container*
Webis uses Enroot containers linked to the repository to facilitate code execution independently of your affiliation to the project.
The container image is built and pushed to the Webis registry by the project maintainer.
The image is specified in the bash script.

## 🐳 Pushing images*
This project's image namespace is `registry.webis.de/code-teaching/theses/artificial-authorship-verification:latest`.
First, you need to log in to the Webis registry using the following command:

```bash
docker login registry.webis.de
```

Then, you can build and push the image using the following command:
`docker compose build --push` builds the image and pushes it to the Webis registry (using `docker-compose.yaml`).
Instructions followed during the building process are specified in the `Dockerfile` and `docker-compose.yaml`.

View image at registry at [GitLab project](https://git.webis.de/code-research/theses/artificial-authorship-verification/container_registry/1461).
