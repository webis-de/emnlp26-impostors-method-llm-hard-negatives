# artificial-authorship-verification
This repository contains the code associated with my Master thesis about authorship verification of artificial-generated texts and human-authored texts.


# Datasets

## Add a new dataset
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
{"id": "1", "text": "This is the first text.", "same": "True", "authors": ["Author1", "Author1"]}
{"id": "2", "text": "This is the second text.", "same": "False", "authors": ["Author3"]}
```
To convert a dataset to a Hugging Face dataset, you can use the `genai_detection/dataset_util.py` script. 
Create a class that inherits from `BaseDatasetLoader` or `Pan23DatasetLoader` and implements the missing methods (like `Pan20DatasetLoader`).
Create a `run_<your-dataset>` function.
Run the `run_<your-dataset>` function from main at the bottom of the script.
This script will read the `pairs.jsonl` and `truth.jsonl` files and convert them into a Hugging Face dataset format.


# Enroot Container
Webis uses Enroot containers linked to the repository to facilitate code execution independently of your affiliation to the project.
The container image is built and pushed to the Webis registry by the project maintainer.
The image is specified in the bash script.

## Pushing images
This project's image namespace is `registry.webis.de/code-research/theses/artificial-authorship-verification:latest`.
`docker compose build --push` builds the image and pushes it to the Webis registry (using `docker-compose.yaml`).
Instructions followed during the building process are specified in the `Dockerfile` and `docker-compose.yaml`.

View image at registry at [gitlab project](https://git.webis.de/code-research/theses/artificial-authorship-verification/container_registry/1461).
