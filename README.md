# Mosaic: H&E Whole Slide Image Cancer Subtype and Biomarker Inference

Mosaic is a deep learning model designed for predicting cancer subtypes and biomarkers from Hematoxylin and Eosin (H&E) stained whole slide images (WSIs). This repository provides the code, pre-trained models, and instructions to use Mosaic for your own datasets.

## Table of Contents

- [System Requirements](#system-requirements)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Initial Setup](#initial-setup)
  - [Web Application](#web-application)
  - [Command Line Interface](#command-line-interface)
  - [Notes](#notes)
- [Output Files](#output-files)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [CSV File Format](#csv-file-format)
- [Cancer Subtypes](#cancer-subtypes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

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
git clone https://github.com/pathology-data-mining/mosaic.git
cd mosaic
uv sync
```

Alternatively, install directly from the repository:

```bash
uv pip install git+https://github.com/pathology-data-mining/mosaic.git
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
mosaic
```

It will start a web server on port 7860 by default. You can access the web interface by navigating to `http://localhost:7860` in your web browser.

### Command Line Interface

To process a single WSI, use the following command:

```bash
mosaic --slide-path /path/to/your/wsi.svs --output-dir /path/to/output/directory
```

To process a batch of WSIs, use:

```bash
mosaic --slide-csv /path/to/your/wsi_list.csv --output-dir /path/to/output/directory
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
mosaic --help
```

If setting port to run in server mode, you may check for available ports using `ss -tuln | grep :PORT` where PORT is the port number you want to check. No output indicates the port may be available. If port is available, set environment variable `export GRADIO_SERVER_PORT="PORT"`

### Notes

- The first time you run the application, it will download the necessary models from HuggingFace. This may take some time depending on your internet connection.
- The models are downloaded to a directory named `data` relative to where you run the application.

## Output Files

### Single Slide Processing

When processing a single slide, the following files are generated in the output directory:

- `{slide_name}_mask.png`: Visualization of the tissue segmentation
- `{slide_name}_aeon_results.csv`: Cancer subtype predictions with confidence scores (if cancer subtype was set to "Unknown")
- `{slide_name}_paladin_results.csv`: Biomarker predictions for the slide

### Batch Processing

When processing multiple slides, in addition to individual slide outputs, combined results are generated:

- `combined_aeon_results.csv`: Cancer subtype predictions for all slides in a single file
- `combined_paladin_results.csv`: Biomarker predictions for all slides in a single file

## Examples

### Example 1: Process a single slide with unknown cancer type

```bash
mosaic --slide-path /data/slides/sample.svs \
       --output-dir /data/results \
       --site-type Primary \
       --cancer-subtype Unknown \
       --segmentation-config Resection
```

### Example 2: Process a single breast cancer slide with known IHC subtype

```bash
mosaic --slide-path /data/slides/breast_sample.svs \
       --output-dir /data/results \
       --site-type Primary \
       --cancer-subtype BRCA \
       --ihc-subtype "HR+/HER2-" \
       --segmentation-config Biopsy
```

### Example 3: Process multiple slides from CSV

Create a CSV file `slides.csv` with the following format:

```csv
Slide,Site Type,Cancer Subtype,Segmentation Config,IHC Subtype
/data/slides/sample1.svs,Primary,Unknown,Resection,
/data/slides/sample2.svs,Metastatic,LUAD,Biopsy,
/data/slides/sample3.svs,Primary,BRCA,TCGA,HR+/HER2-
```

Then run:

```bash
mosaic --slide-csv slides.csv --output-dir /data/results
```

## Advanced Usage

### Adjusting Performance

You can control the number of workers for feature extraction to balance between speed and memory usage:

```bash
mosaic --slide-path /path/to/slide.svs \
       --output-dir /path/to/output \
       --num-workers 8
```

### Running in Server Mode

To run Mosaic as a web server accessible from other machines:

```bash
export GRADIO_SERVER_PORT=7860
mosaic --server-name 0.0.0.0 --server-port 7860
```

Check for available ports using:
```bash
ss -tuln | grep :7860
```

To share the application publicly (use with caution):

```bash
mosaic --share
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
mosaic --debug
```

This will create a `debug.log` file with detailed information about the processing steps.

## CSV File Format

When processing multiple slides using the `--slide-csv` option, the CSV file must contain the following columns:

### Required Columns

- **Slide**: Full path to the WSI file (e.g., `/path/to/slide.svs`)
- **Site Type**: Either `Primary` or `Metastatic`

### Optional Columns

- **Cancer Subtype**: OncoTree code for the cancer subtype (e.g., `LUAD`, `BRCA`, `COAD`). Use `Unknown` to let Aeon infer the cancer type.
- **Segmentation Config**: One of `Biopsy`, `Resection`, or `TCGA`. Defaults to `Biopsy` if not specified.
- **IHC Subtype**: For breast cancer (BRCA) only. One of:
  - `HR+/HER2+`
  - `HR+/HER2-`
  - `HR-/HER2+`
  - `HR-/HER2-`

### CSV Example

```csv
Slide,Site Type,Cancer Subtype,Segmentation Config,IHC Subtype
/data/slides/lung1.svs,Primary,LUAD,Resection,
/data/slides/breast1.svs,Primary,BRCA,Biopsy,HR+/HER2-
/data/slides/unknown1.svs,Metastatic,Unknown,TCGA,
```

## Cancer Subtypes

Mosaic uses OncoTree codes to identify cancer subtypes. Common examples include:

- **LUAD**: Lung Adenocarcinoma
- **LUSC**: Lung Squamous Cell Carcinoma
- **BRCA**: Breast Invasive Carcinoma
- **COAD**: Colon Adenocarcinoma
- **READ**: Rectal Adenocarcinoma
- **PRAD**: Prostate Adenocarcinoma
- **SKCM**: Skin Cutaneous Melanoma

For a complete list of supported cancer subtypes, see the [OncoTree website](http://oncotree.mskcc.org/).

When the cancer subtype is set to `Unknown`, Mosaic will use the Aeon model to predict the most likely cancer subtype based on the H&E image features.

## Troubleshooting

### HuggingFace Authentication Errors

If you encounter authentication errors when downloading models:

1. Ensure you have access to the PDM-Group on HuggingFace
2. Create a HuggingFace access token with appropriate permissions
3. Set the `HF_TOKEN` environment variable correctly

### Out of Memory Errors

If you encounter GPU out-of-memory errors:

1. Reduce the number of workers: `--num-workers 2`
2. Process slides sequentially instead of in batch
3. Consider using a GPU with more memory

### Tissue Segmentation Issues

If tissue is not being detected correctly:

1. Try a different segmentation configuration (`Biopsy`, `Resection`, or `TCGA`)
2. Check that the slide file is not corrupted
3. Verify the slide format is supported (e.g., `.svs`, `.tif`)

### Port Already in Use

If the default port 7860 is already in use:

1. Check for running processes: `ss -tuln | grep :7860`
2. Use a different port: `export GRADIO_SERVER_PORT=7861`
3. Or specify the port directly: `mosaic --server-port 7861`

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the terms specified in the LICENSE file.
