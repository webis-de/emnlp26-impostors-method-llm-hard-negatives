# LLM-based Impostor Generation & Reframing the Impostors Method as a Hypothesis Test
This repository contains the code associated with ARR May submission "The Impostors Method as a Statistical Test for Authorship Verification
with Large Language Models as Sampling Source for Hard Negatives" about authorship verification of human-authored texts.
We (1) reinterpret the Impostors Method for authorship verification as a statistical hypothesis test and (2) 
investigate LLM-generated texts as controllable in-domain impostors. 
We re-implement the orginal Impostors Method by 
[Koppel and Winter (2014)](https://asistdl.onlinelibrary.wiley.com/doi/pdf/10.1002/asi.22954) 
as well as several of its existing variants and extensions.

---
## Impostors Method
You may find the re-implementaion of the original Impostors Method in the `genai_detection/detectors` directory.
This directory also contains some additional baseline detectors (e.g., PPMd, Unmasking, etc.).
Traditional mpostor generation techniques are implemented in the `genai_detection/impostor_generators` directory, while LLM-based paraphrasing techniques are implemented in the `
genai_detection/paraphrasers` directory.

### Retrieval-based Impostor Generation
We use (1) the Google Search API via SerpAPI, (2) ChatNoir, (3) Startpage to retrieve on-topic impostors from the web.

### Paraphrases
We use LLMs to generate paraphrases of texts as impostors.
We compare prompted open-source models hosted by GWDG with two-step OpenAI GPT5 Nano for impostor generation.

#### DSPy
[DSPy](https://dspy.ai/) is a declarative framework for building modular AI software.
DSPy is short for _Declarative Self-improving Python_.
When designing AI systems with LLMs, one often has to deal with prompt engineering and choosing the right model or prompting strategy.
DSPy makes LLMs easily interchangeable and omits the need for prompting according to the LLMs preferences.
Instead of engineering prompts directly, one uses structured and declarative natural language:
Every AI component (i.e., component that interacts with LLMs) needs a _signature_ (i.e., input and output parameter specification) and a _module_ (e.g., `Predict`) to invoke the LM based on the signature.
DSPy expands the signatures into prompts automatically using the input, output fields and the docstring.
[DSPy](https://arxiv.org/abs/2310.03714) is the second version of DSP, which was developed at Stanford University.
Find more papers about DSPy [here](https://github.com/stanfordnlp/dspy?tab=readme-ov-file#-citation--reading-more).

### OpenAI
Remember checking [billing](https://platform.openai.com/settings/organization/billing/overview) and 
[API usage](https://platform.openai.com/settings/organization/usage).

---
## Extensions of the Impostors Method
You may find the extensions to the Impostors Method in the `genai_detection/ablations` directory.
We re-implemented:
- [Khonji & Iraqi (2014)](https://downloads.webis.de/pan/publications/papers/khonji_2014.pdf)
- [Gutierrez et al. (2015)](https://ceur-ws.org/Vol-1391/74-CR.pdf)
- [Kestemont et al. (2016)](https://linkinghub.elsevier.com/retrieve/pii/S0957417416303116)
- [Potha & Stamatatos (2017)](https://link.springer.com/chapter/10.1007/978-3-319-65813-1_14)
- [Nagy (2024)](https://ceur-ws.org/Vol-3834/paper61.pdf)


---
## `Scripts` Directory
The `scripts` directory contains convenience scripts including generating LaTex tables for results, generating 
certain paraphrases.

---
## Impostor UI
The UI in `ui/impostor_ui` is not the focus of this repository, but still a work in progress.

Our experiments do not depend on the UI, but are run via 
`genai_detection/experiments/reproduction/run_prec_recall_curves.py`. 

Note: The mongoDB needs to be populated with the text pairs for the experiments to run.

---
## Results
Once you run scripts and experiments, you can find the results in the `results` directory.

---

## Getting Started

The project uses [**Poetry**](https://python-poetry.org/) for dependency management and packaging. 
[**Poetry**](https://python-poetry.org/) is a modern Python tool for **dependency management** and **packaging**. 
It replaces tools like `pip`, `virtualenv`, and `setuptools` with a single, streamlined workflow.

#### Key Benefits

- Manages dependencies via `pyproject.toml`
- Automatically creates isolated virtual environments
- Simplifies package building and publishing
- Provides a clean CLI for common tasks (`install`, `add`, `build`, etc.)


### Requirements

- **Python ≥3.10 and <3.13**
  - Word Mover's Distance (WMD) requires Python <3.13, I use Python 3.11.9
- **Poetry ≥1.3**
- **Torch**
  - MacOS does not work with torch's cpu wheel. Ensure you have `torch = { version = "^2.5.0", source = "pypi" }
  torchvision = { version = "^0.20.0", source = "pypi" }
  torchaudio = { version = "^2.5.0", source = "pypi" }` in your `pyproject.toml` to install from PyPI instead of the default source.

Install poetry using the following command for MacOS:

```bash
brew install poetry
```

For the paraphrasers used in the impostor generation, you need to set up the following API keys in your `.env` file:
- [DEEPL_KEY](https://www.deepl.com/) (for Translation via DeepL: 500,000 characters/month in free plan)
- [SAIA_KEY](https://services.kisski.de/services/en/service/?service=2-02-llm-service.json) (for SAIA paraphraser 
  models hosted by [GWDG](https://gwdg.de/))
- [SERPAPI_KEY](https://serpapi.com/users/sign_up) (Optional: for Google search API: 250 queries/month in free plan)

### Set up the project
1. Clone the repository
2. Navigate to the project directory: `cd artificial-authorship-verification`
3. Install the dependencies using Poetry:
   ```bash
   poetry install
   ```

### Common Poetry Commands
- Add a new dependency:
  ```bash
  poetry add <package-name>
  ```
- Update dependencies:
  ```bash
  poetry update
  ```
---

# Datasets

Before you can run the code, you need to download the datasets and place them in the `data/datasets` directory.
The datasets need to be converted to the Hugging Face dataset format.
After downloading the datasets defined in the following, and adding them as described below, you can use the 
provided script `genai_detection/dataset/dataset_util.py` to convert them.
Note that you have to comment dataset methods you do not need at the bottom of the script, and run the script to convert the datasets.

## Current datasets
- **Koppel et al. (2014)**: [Blog Authorship Corpus](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus?resource=download)
  - Contains blog posts with authorship information.
  - The dataset is in CSV format.
- **Koppel et al. (2014) Student Essays**: not publicly available
  - Contains student essays with authorship information.
- **Gutenberg**: [Gutenberg dataset](https://www.gutenberg.org/)
  - Contains texts from the Gutenberg project with authorship information.
  - The dataset is in TXT format.

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
{"id": "1", "pair": ["This is the first text.","This is the second text."], "same": "True", "authors": ["Author1", "Author1"]}
{"id": "2", "pair": ["This is the first text.","This is the second text."], "same": "False", "authors": ["Author3", "Author4"]}
```
To convert a dataset to a Hugging Face dataset, you can use the `genai_detection/dataset_util.py` script. 
Create a class that inherits from `BaseDatasetLoader` and implements the missing methods (like `BlogCorpusDatasetLoader`).
Create a `run_<your-dataset>` function.
Run the `run_<your-dataset>` function from main at the bottom of the script.
This script will read the `pairs.jsonl` and `truth.jsonl` files and convert them into a Hugging Face dataset format.

---
## Mongo DB
The project uses a MongoDB database to store text pairs used in experiments, paraphrased texts, scores outputted and 
their paraphrase scores, etc.

The database is located on the Webis Kubernetes cluster.

### Kubernetes
We used [Kubernetes](https://kubernetes.io/) to deploy the MongoDB database on the Webis cluster.
- [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) are the smallest deployable units of computing that you can create and manage in Kubernetes.
A pod is a group of **one or more containers**, with shared storage and network resources, and a specification for how to run the containers.
- A [container](https://kubernetes.io/docs/concepts/containers/) packages an application along with its runtime dependencies. 
Containers in a pod access and share data via [volumes](https://kubernetes.io/docs/concepts/storage/volumes/).
- Volumes can be of different types; enabling filesystem sharing between containers in the same or across different pods, enabling durable storing data (availability even if Pod restarts: **persistent volumes**), limiting data access to read-only, etc.
A Pod can use a number of volume types simultaneously.


### Initial Setup of the Database
We used [helm](https://helm.sh/) to set up the MongoDB database on the Kubernetes cluster.
Helm generates YAML template files to deploy the database (instead of manually writing or copying them).
Helm uses a packaging format called [charts](https://helm.sh/docs/topics/charts/).
- A chart is used to deploy something. It is a collection of files in a directory. The directory name is the name of the chart. The files describe a related set of Kubernetes resources. The files are laid out in a directory tree. The charts are located in a [chart repository](https://helm.sh/docs/topics/chart_repository/).

We found a [MongoDB helm chart](https://artifacthub.io/packages/helm/bitnami/mongodb) that we customized to our needs via the `values.yaml` in the `helm` directory (e.g., bigger persistent volume size (i.e. 16 GB instead of 8 GB) on Betaweb (i.e. `csi-rbd-retain`, where [retain means persistent](https://kb.webis.de/k8s-manual/ceph-in-k8s.html#provision-a-cephfs-subvolume) and the absence of `ssd` means not gammaweb or no GPU), architecture: `standalone` ~~replicaset (i.e. PersistentSet with multiple Pods)~~, with recreation of pods when they fail (i.e.`Recreate`, not keeping old pod until new one is set up because volume is not shared), and without a networkPolicy because that is not done at Webis).

The command to **upgrade/install the chart (Deployment at webis)** is:
```bash
helm upgrade --install --namespace webisservices --create-namespace artificial-authorship-verification oci://registry-1.docker.io/bitnamicharts/mongodb -f values.yaml --set auth.rootPassword="SecurePassword123!"
```
- `--install`: Install the chart if it is not already installed.
- `--namespace webisservices`: Specifies the namespace in which to install the chart. Only `webisservices`or `webisstud` is allowed for Webis projects.
- `--create-namespace`: Creates the namespace if it does not already exist.
  - Webis namespaces have certain logic (refer to other as examples via `kubectl get namespaces`).
- `artificial-authorship-verification`: Name of the release.
- `oci://registry-1.docker.io/bitnamicharts/mongodb`: Location of the chart.
- `-f values.yaml`: Specifies the values file to use for the installation. It overrides the default values provided by the chart.
- `--set auth.rootPassword="SecurePassword123!"`: Sets the root password for the MongoDB database. You can change it to a secure password of your choice. _SecurePassword123!_ is just an example; I used a different password.

The command to **generate the template file without installing it** is:
```bash
helm template --namespace webisservices --create-namespace artificial-authorship-verification oci://registry-1.docker.io/bitnamicharts/mongodb -f values.yaml --set auth.rootPassword="SecurePassword123!" > "template.k8s.yaml"
```
- This command generates the [Kubernetes YAML template file](https://helm.sh/docs/chart_template_guide/debugging/) and saves it as `template.k8s.yaml`.
- You can then review the file inspecting settings like password, volume size, etc. Different original files are separated by `---` in the YAML file.
- You can also directly edit the file before applying it to the cluster using `kubectl apply -f template.k8s.yaml`.

You may find the database at the following address:
`artificial-authorship-verification-mongodb.webisservices.svc.cluster.local`

### Accessing the Database via kubectl mongosh client
You can access it using the `mongosh` client, to directly interact with the database:
```bash
kubectl run --namespace webisservices artificial-authorship-verification-mongodb-client --rm --tty -i --restart='Never' --env="MONGODB_ROOT_PASSWORD=SecurePassword123!" --image registry-1.docker.io/bitnami/mongodb:latest --command -- bash
```
- `-rm`: Remove the pod after exiting.
- `-tty -i`: Interactive terminal.
- `--restart='Never'`: Do not restart the pod after it exits.
- `--env="MONGODB_ROOT_PASSWORD=$MONGODB_ROOT_PASSWORD"`: Set the environment variable for the root password.
- `--image registry-1.docker.io/bitnami/mongodb:latest`: Use the Bitnami MongoDB image.
- `--command -- bash`: Run the bash shell.
This will open a bash shell in the pod.
You first have to log in with:
```bash
mongosh admin --host "artificial-authorship-verification-mongodb" --authenticationDatabase admin -u root -p $MONGODB_ROOT_PASSWORD
```
- `admin`: The database to connect to.
- `--host "artificial-authorship-verification-mongodb"`: The host address of the MongoDB database.
- `--authenticationDatabase admin`: The authentication database.
- `-u root`: The username.
- `-p $MONGODB_ROOT_PASSWORD`: The password.
From there, you can interact with the MongoDB database using `mongosh`:
- create a database: `use mydatabase`
- show databases: `show dbs`
- show collections: `show collections`
- insert a document: `db.mycollection.insertOne({name: "Name", age: 45})`
- find documents: `db.mycollection.find()`

Otherwise you can connect to it from anywhere on the Kubernetes cluster using the address above.
You cannot access it from Gammaweb.


### Change password of the database
You need to (1) edit the secret in Kubernetes that stores the password locally and (2) since the password is also stored in the mongoDB, you need to change it there as well.
For (1):
```bash
kubectl edit secret -n webisservices artificial-authorship-verification-mongodb
```
1.1. Locally find a new password and [convert it to base64](https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-config-file/): `echo -n "newpassword" | base64`

1.2. Insert the base64 encoded password in the `data` section under `mongodb-root-password:` that you find when editing the secret.

For (2):
Inside the mongoDB shell (c.f. above), run:
```bash
db.changeUserPassword("root", "newpassword")
```
- The user is `root`.

You may now need to restart the pod again to make sure everything works with the new password:
(1) `kubectl get pods -n webisservices`, find the name of you pod (without client in the name), then (2) `kubectl delete pod <pod-name> -n webisservices` so that the pod will restart and the password change is applied.


### No Disk Space Left for mongodb
Find out whether the issue is space:
```bash 
kubectl logs artificial-authorship-verification-mongodb-566757dd9b-49pf7 -n webisservices
```
Increase the size of the persistent volume (and alter the requested size):
```bash
kubectl edit pvc -n webisservices artificial-authorship-verification-mongodb
```
To finalize the fix, renew deployment:
```bash
kubectl rollout restart -n webisservices deployment/artificial-authorship-verification-mongodb
```
Also, you may need to restart the path forwarding.

### Populate the database with original texts
You can populate the database with paraphrased texts and their evaluation scores using the scripts provided in the 
`genai_detection/dataset` directory.
When running 
```bash
python3 genai_detection/dataset/dataset_util.py
``` 
locally, with the correct `run_DATASET_NAME` function, you will populate the database with the text pairs.

### Part forwarding

Via terminal:
```bash
kubectl port-forward -n webisservices deployment/artificial-authorship-verification-mongodb 27018:27017
```
- This will forward the local port `27018` to the remote port `27017` of the MongoDB deployment.
- You can then access the database locally at `localhost:27018`.

If you want automatic retries in case of errors, run:
```bash
while ! kubectl port-forward -n webisservices deployment/artificial-authorship-verification-mongodb 27018:27017; do sleep 1; done
```
This will rerun the command in case it breaks (because your WIFI is turned off or something else happened).
You can stop it via `Ctrl + C`.



