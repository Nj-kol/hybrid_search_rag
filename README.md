# Hybrid Search RAG

## Sample Conversations

### Text to Text

### Voice to Text


## Setup

**Prerequisites**

- Docker
- Conda/Miniconda

### Components

**Architecture Components**

The following components needs to be installed for this
- Ollama ( For Serving LLM models )
- Qdrant ( Vector Database )
- Open WebUI ( Chatbot UI )

**Ollama Setup**

Install Ollama : See https://ollama.com/

Pull an LLM Model

```shell
ollama pull llama3.1:8b
```

Note: For this excercise, please use a model which supports tool calling.

**Qdrant VectorDB**

See : [Qdrant Vector DB](docs/VectorDB.md)

**Open WebUI**

See : [Open WebUI](docs/OpenWebUI.md)

### Code

**Prerequisites**

Create a new python environment & install dependencies

```shell
## Create new enviroment
conda create -n gen_ai_demo python=3.11

## Activate the environment
conda activate gen_ai_demos

## Intall the dependencies 
pip install -r requirements.txt
```

Launch JupyterLab `jupyter lab`

**Step 1: Pre-download all embedding models locally to a specific path**

Run the notebook `Download_Embedding_Models.ipynb`

**Step 2: Ingest Data to Vector DB**

Before running this notebook, ensure that Qdrant DB is running.

Run the notebook `Qdrant_Hybrid_ETL.ipynb`

**Step 3: Install the OpenWeb UI Piplelines Project**

For hosting & serving our app, we shall be using the [Pipelines](https://docs.openwebui.com/pipelines) project, to make it easy to integate withe fantastic [OpenWebUI](https://openwebui.com/) frontend.

Clone the repo: `git clone https://github.com/open-webui/pipelines.git`

Copy the `apps` folder from this project to the root folder of the `pipelines` project. So the directory structure would look something like :

```shell
.
├── CONTRIBUTING.md
├── Dockerfile
├── LICENSE
├── README.md
├── apps
...
├── main.py
├── pipelines
...
```

Then copy the file `pipelines` under the `pipelines` folder.

Finally, start the pipelines server :

```shell
## Activate the enviromnent
conda activate gen_ai_demos

## For Windows
start.bat

## For Linux
bash ./start.sh
```

**Step 4: Integrate the Open WebUI and Pipelines**

## UIs

- Qdrant Vector store: http://localhost:6333/dashboard
- Open Web UI :  http://localhost:3000/

## References

- https://qdrant.tech/documentation/search-precision/reranking-hybrid-search/
