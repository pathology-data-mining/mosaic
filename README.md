# Mosaic: H&E Whole Slide Image Cancer Subtype and Biomarker Inference

Mosaic is a deep learning model designed for predicting cancer subtypes and biomarkers from Hematoxylin and Eosin (H&E) stained whole slide images (WSIs). This repository provides the code, pre-trained models, and instructions to use Mosaic for your own datasets.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

### System requirements

Supported systems:

- Linux (x86) with GPU (NVIDIA CUDA)

### Pre-requisites

- [python3.11+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/)

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

## Installation

```bash
uv pip install git+ssh://git@github.com/pathology-data-mining/paladin_webapp.git@dev
```

## Usage

### Initial Setup

<b>NOTE</b>: In order to run this app, the user needs to be added to the [PDM Group](https://huggingface.co/PDM-Group) and the user needs to set the following environment variable. The token may be obtained from clicking on the user icon on the top right of the HuggingFace website and selecting "Access Tokens". When creating the token, select all read options for your private space and the PDM-Group space.

```bash
export HF_TOKEN="TOKEN-FROM-HUGGINGFACE"
```

Additionally, set the location for huggingface home where models and other data from HuggingFace may be downloaded.

```bash
export HF_HOME="PATH-TO-HUGGINGFACE-HOME"
```

### Web Application

Run the web application with:

```bash
mosaic_app
```

It will start a web server on port 7860 by default. You can access the web interface by navigating to `http://localhost:7860` in your web browser.

### Command Line Interface

To process a WSI, use the following command:

```bash
mosaic_app --slide-path /path/to/your/wsi.svs --output-dir /path/to/output/directory
```

To process a batch of WSIs, use:

```bash
mosaic_app --slide-csv /path/to/your/wsi_list.csv --output-dir /path/to/output/directory
```

The CSV file should at least contain columns `Slide`, and `Site Type`.
Optionally, it can also contain `Cancer Subtype`, `Segmentation Config`, and `IHC Subtype`.

- `Slide` should contain the full path to the WSI file.
- `Site Type` should be one of `Primary`, or `Metastatic`.
- `Cancer Subtype` should be the oncotree code for the cancer subtype.
- `Segmentation Config` should be one of `Biopsy`, `Resection`, or `TCGA`.
- `IHC Subtype` should be one of `HR+/HER2+`, `HR+/HER2-`, `HR-/HER2+`, or `HR-/HER2-`.

See additional options with the help command. This command may take a few seconds to run:

```bash
mosaic_app --help
```

If setting port to run in server mode, you may check for available ports using `ss -tuln | grep :PORT` where PORT is the port number you want to check. No output indicates the port may be available. If port is available, set environment variable `export GRADIO_SERVER_PORT="PORT"`

### Notes

- The first time you run the application, it will download the necessary models from HuggingFace. This may take some time depending on your internet connection.
- The models are downloaded to a directory relative to where you run the application. (A subdirectory named `data`).
