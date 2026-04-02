---
title: Mosaic
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: apache-2.0
hf_oauth: true
---

# Mosaic: H&E Whole Slide Image Cancer Subtype and Biomarker Inference

Mosaic is a deep learning model designed for predicting cancer subtypes and biomarkers from Hematoxylin and Eosin (H&E) stained whole slide images (WSIs). This repository provides the code, pre-trained models, and instructions to use Mosaic for your own datasets.

## Table of Contents

- [System Requirements](#system-requirements)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces)
- [Usage](#usage)
  - [Initial Setup](#initial-setup)
  - [Web Application](#web-application)
  - [Command Line Interface](#command-line-interface)
  - [Notes](#notes)
- [Output Files](#output-files)
- [Examples](#examples)
- [Advanced Usage](#advanced-usage)
- [User Storage Management (HF Spaces)](#user-storage-management-hf-spaces)
  - [My Files Tab](#my-files-tab)
  - [My Results Tab](#my-results-tab)
  - [Storage Quotas](#storage-quotas)
  - [Local Debug Mode](#local-debug-mode)
- [CSV File Format](#csv-file-format)
- [Cancer Subtypes](#cancer-subtypes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Architecture](#architecture)
- [License](#license)

### System requirements

Supported systems:

- Linux (x86) with GPU (NVIDIA CUDA)

### Pre-requisites

- [python3.11](https://www.python.org/)
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

Note that when installing via `uv sync`, the virtual environment will be created in the `./.venv` directory. To activate it, run:

```bash
source .venv/bin/activate
```

Alternatively, create a virtual environment mosaic-venv (in a subdirectory), activate it, and install the app directly from the repository:

```bash
uv venv mosaic-venv --python 3.11
source mosaic-venv/bin/activate
uv pip install git+https://github.com/kmboehm/paladin.git@main
```

## Deploying to Hugging Face Spaces

This repository is configured for deployment on Hugging Face Spaces with Zero GPU support.

### Prerequisites

1. You need to be added to the [PDM Group](https://huggingface.co/PDM-Group) on Hugging Face to access the models
2. Create a Hugging Face access token with read permissions for the PDM-Group space

### Deployment Steps

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Choose "Zero GPU" as the hardware option (if available)
4. Clone this repository to your Space or push the code
5. In your Space settings, add a secret named `HF_TOKEN` with your Hugging Face access token
6. The app will automatically start and download the necessary models on first run

### Zero GPU Configuration

The app uses the `@spaces.GPU` decorator to allocate GPU resources only when needed for inference. This allows efficient use of Zero GPU resources on Hugging Face Spaces. The GPU is automatically allocated when:

- Processing tissue segmentation
- Extracting features with CTransPath and Optimus models
- Running Aeon and Paladin model inference

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

#### Complete CLI Options Reference

##### Processing Options

- `--slide-path PATH`: Path to a single slide for processing (mutually exclusive with `--slide-csv`)
- `--slide-csv PATH`: CSV file with slide settings for batch processing (see [CSV File Format](#csv-file-format))
- `--output-dir PATH`: Directory to save output results (required for CLI processing)

##### Single Slide Parameters

These options apply when using `--slide-path` for single slide processing:

- `--site-type {Primary,Metastatic}`: Site type of the slide (default: `Primary`)
- `--cancer-subtype CODE`: Cancer subtype OncoTree code (default: `Unknown` to infer with Aeon model)
- `--segmentation-config {Biopsy,Resection,TCGA}`: Tissue segmentation configuration (default: `Biopsy`)
- `--ihc-subtype SUBTYPE`: IHC subtype for breast cancer (BRCA) only. Options:
  - `HR+/HER2+`
  - `HR+/HER2-`
  - `HR-/HER2+`
  - `HR-/HER2-`
- `--sex {Male,Female,Unknown}`: Patient sex for improved Aeon inference (default: `Unknown`)
- `--tissue-site SITE`: Primary tissue site for improved Aeon inference (default: `Unknown`)
  - Examples: `Lung`, `Breast`, `Colon`, `Liver`, `Brain`, `Lymph Node`, `Bone`
  - See `data/tissue_site_original_to_idx.csv` for complete list

##### Performance & Processing

- `--num-workers N`: Number of workers for feature extraction (default: 4)
  - Increase for faster processing (e.g., 8-16) if you have sufficient CPU/memory
  - Decrease (e.g., 2-4) if encountering memory issues

##### Model Management

- `--skip-model-download`: Skip downloading models from HuggingFace (assumes models are already cached)
- `--download-models-only`: Download models from HuggingFace and exit without running analysis

##### Web Server Options

- `--server-name ADDRESS`: Server address for Gradio web interface (default: `0.0.0.0`)
- `--server-port PORT`: Server port for Gradio web interface (default: uses `GRADIO_SERVER_PORT` env var or 7860)
- `--share`: Create a public shareable link for the Gradio interface (use with caution)

##### Debugging

- `--debug`: Enable debug logging (creates `debug.log` file with detailed information)

##### Getting Help

See all available options with:

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
       --segmentation-config Resection \
       --sex Female \
       --tissue-site Lung
```

### Example 2: Process a single breast cancer slide with known IHC subtype

```bash
mosaic --slide-path /data/slides/breast_sample.svs \
       --output-dir /data/results \
       --site-type Primary \
       --cancer-subtype BRCA \
       --ihc-subtype "HR+/HER2-" \
       --segmentation-config Biopsy \
       --sex Female \
       --tissue-site Breast
```

### Example 3: Process multiple slides from CSV

Create a CSV file `slides.csv` with the following format:

```csv
Slide,Site Type,Cancer Subtype,Segmentation Config,IHC Subtype,Sex,Tissue Site
/data/slides/sample1.svs,Primary,Unknown,Resection,,Female,Lung
/data/slides/sample2.svs,Metastatic,LUAD,Biopsy,,,Liver
/data/slides/sample3.svs,Primary,BRCA,TCGA,HR+/HER2-,Female,Breast
```

Then run:

```bash
mosaic --slide-csv slides.csv --output-dir /data/results
```

## Advanced Usage

### Model Management

#### Download Models Before Processing

To download models from HuggingFace without running any analysis:

```bash
mosaic --download-models-only
```

Or using the Makefile:

```bash
make download-models
```

#### Skip Model Download

If models are already cached and you want to skip the download check:

```bash
mosaic --skip-model-download --slide-path /path/to/slide.svs --output-dir /path/to/output
```

This is useful for offline processing or when you know models are already cached.

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

## User Storage Management (HF Spaces)

When running Mosaic on HuggingFace Spaces, logged-in users have access to ephemeral file storage for uploaded slides and analysis results. This feature allows you to:

- **Re-analyze slides** with different settings without re-uploading
- **View previous analysis results** from past sessions
- **Download results** at any time during your session

**Important**: All stored files are **ephemeral** and will be deleted when the HuggingFace Spaces instance restarts. This typically happens during updates or when the instance is idle for extended periods.

### My Files Tab

The **My Files** tab (visible only when logged in) provides access to your uploaded slides:

**Features:**
- **Storage usage display**: Shows current usage vs. quota (e.g., "2.3 GB / 5 GB")
- **Color-coded warnings**:
  - 💾 Normal: < 80% quota used
  - ⚠️ Warning: 80-99% quota used (delete old files to free space)
  - ⛔ Error: ≥ 100% quota exceeded (upload blocked until space freed)
- **File browser**: View all uploaded slides with:
  - Slide ID (unique identifier)
  - Original filename
  - File size
  - Upload date
  - Number of analyses performed
- **File actions**:
  - **Download**: Download original slide file
  - **Delete**: Remove slide and all associated analysis results
  - **Refresh**: Update file list

**Typical workflow:**
1. Upload slides via main analysis tab
2. Review uploaded files in My Files tab
3. Delete old slides when approaching quota limit

### My Results Tab

The **My Results** tab (visible only when logged in) displays all analysis results from your session:

**Features:**
- **Results browser**: View all analyses with:
  - Analysis ID (unique identifier)
  - Slide name
  - Analysis date/time
  - Predicted cancer subtype
  - Analysis settings (sex, tissue site, site type)
- **Result viewer**: Select an analysis to view:
  - Full metadata (settings, timestamps)
  - Tissue segmentation mask (PNG)
  - Aeon predictions (top cancer subtypes with confidence scores)
  - Paladin biomarker predictions (if applicable)
- **Result actions**:
  - **View**: Display full analysis details
  - **Download ZIP**: Download all results as a ZIP file
  - **Delete**: Remove specific analysis result
  - **Refresh**: Update results list

**Download ZIP contents:**
```
{analysis_id}.zip
├── metadata.json          # Analysis settings and timestamps
├── slide_mask.png         # Tissue segmentation visualization
├── {analysis_id}_aeon_results.csv      # Cancer subtype predictions
└── {analysis_id}_paladin_results.csv   # Biomarker predictions (if available)
```

### Storage Quotas

**Per-user quota**: 5 GB (default)

This limit is enforced to prevent disk exhaustion on shared HuggingFace Spaces instances. When you approach or exceed your quota:

1. **Automatic cleanup**: Oldest files are deleted automatically (FIFO - First In, First Out)
2. **Manual cleanup**: You can delete files manually via the My Files tab
3. **Upload blocking**: New uploads are blocked at 100% quota until space is freed

**Typical storage usage:**
- Small WSI (biopsy): 100-300 MB
- Medium WSI (tissue section): 500 MB - 1 GB
- Large WSI (whole tissue): 1-2 GB
- Analysis results: ~1-2 MB each (negligible)

**Example**: With a 5 GB quota, you can store approximately **5-10 slides** concurrently.

### Local Debug Mode

When running Mosaic **locally** (not on HuggingFace Spaces), the My Files and My Results tabs are still available for debugging:

- All files are stored under a **universal "local_user"** username
- Storage path: `/tmp/mosaic_user_data/local_user/`
- UI shows **🔧 [Local Debug Mode]** indicator
- Files are still ephemeral (cleared on system reboot)
- No authentication required

This mode is useful for:
- Testing the storage feature locally
- Debugging upload/result workflows
- Development and testing

**Enable local debug mode:**
```bash
# Simply run locally (not on HuggingFace Spaces)
make run-ui
# or
mosaic
```

The tabs will automatically detect local mode and show the debug indicator.

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
- **Sex**: Patient sex for improved Aeon cancer subtype inference. One of `Male`, `Female`, or `Unknown`.
- **Tissue Site**: Primary tissue site for improved Aeon cancer subtype inference. Examples include:
  - `Lung`
  - `Breast`
  - `Colon`
  - `Liver`
  - `Brain`
  - `Lymph Node`
  - `Bone`
  - See `data/tissue_site_original_to_idx.csv` for complete list of supported tissue sites.

### CSV Example

```csv
Slide,Site Type,Cancer Subtype,Segmentation Config,IHC Subtype,Sex,Tissue Site
/data/slides/lung1.svs,Primary,LUAD,Resection,,Male,Lung
/data/slides/breast1.svs,Primary,BRCA,Biopsy,HR+/HER2-,Female,Breast
/data/slides/unknown1.svs,Metastatic,Unknown,TCGA,,,Liver
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

## Makefile Commands

This project includes a Makefile with many useful commands for development, testing, and deployment. You can see all available commands by running:

```bash
make help
```

Here are the main Makefile targets:

### Development Setup
- `make install` - Install production dependencies using uv
- `make install-dev` - Install development dependencies using uv

### Testing
- `make test` - Run all tests
- `make test-fast` - Run tests without coverage (faster)
- `make test-coverage` - Run tests with detailed coverage report
- `make test-ui` - Run only UI tests
- `make test-cli` - Run only CLI tests
- `make test-verbose` - Run tests with verbose output and show print statements
- `make test-specific` - Run specific test (usage: `make test-specific TEST=tests/test_cli.py::TestClass::test_method`)
- `make test-watch` - Run tests in watch mode (requires pytest-watch)

### Code Quality
- `make lint` - Run linting checks with pylint
- `make lint-strict` - Run pylint on both src and tests
- `make format` - Format code with black
- `make format-check` - Check code formatting without making changes
- `make quality` - Run all code quality checks

### Application
- `make run-ui` - Launch Gradio web interface
- `make run-ui-public` - Launch Gradio web interface with public sharing
- `make run-single` - Run single slide analysis (usage: `make run-single SLIDE=path/to/slide.svs OUTPUT=output_dir [ARGS="--extra-args"]`)
- `make run-batch` - Run batch analysis from CSV (usage: `make run-batch CSV=settings.csv OUTPUT=output_dir [ARGS="--extra-args"]`)

### Docker
- `make docker-build` - Build Docker image
- `make docker-build-no-cache` - Build Docker image without cache
- `make docker-run` - Run Docker container (web UI mode)
- `make docker-run-cli` - Run Docker container with mosaic CLI (usage: `make docker-run-cli ARGS="--help"`)
- `make docker-run-single` - Run Docker container (single slide mode, usage: `make docker-run-single SLIDE=path/to/slide.svs [ARGS="--extra-args"]`)
- `make docker-run-batch` - Run Docker container (batch mode, usage: `make docker-run-batch CSV=path/to/slides.csv [ARGS="--extra-args"]`)
- `make docker-shell` - Open shell in Docker container
- `make docker-tag` - Tag Docker image for registry
- `make docker-push` - Push Docker image to registry
- `make docker-clean` - Remove Docker image
- `make docker-prune` - Clean up Docker build cache

### Cleanup
- `make clean` - Remove build artifacts and cache files
- `make clean-outputs` - Remove output files (masks, results CSVs)
- `make clean-all` - Remove all build artifacts, cache, and Docker images

### Model Management
- `make download-models` - Download required models from HuggingFace

### Documentation
- `make docs-requirements` - Show what needs to be documented

### CI/CD
- `make ci-test` - Run all CI checks (no lint to save time)
- `make ci-test-strict` - Run all CI checks including pylint
- `make ci-docker` - Build Docker image for CI

### Development Utilities
- `make shell` - Open Python shell with project in path
- `make ipython` - Open IPython shell with project in path
- `make notebook` - Start Jupyter notebook server
- `make check-deps` - Check for outdated dependencies
- `make update-deps` - Update dependencies (be careful!)
- `make lock` - Update lock file

### Git Hooks
- `make pre-commit-install` - Install pre-commit hooks
- `make pre-commit-uninstall` - Uninstall pre-commit hooks

### Information
- `make info` - Display project information
- `make version` - Show version information
- `make tree` - Show project directory tree (requires tree command)

### Performance
- `make profile` - Profile a single slide analysis (usage: `make profile SLIDE=path/to/slide.svs`)
- `make benchmark` - Run performance benchmarks

## Telemetry & Privacy

Mosaic collects anonymous usage telemetry to help improve the tool. This section explains what data is collected and how to opt out.

### What Data is Collected

When running on HuggingFace Spaces, Mosaic collects the following telemetry data:

- **Application events**: App start/shutdown, analysis start/complete, heartbeat
- **Analysis metadata**: Number of slides processed, GPU type, duration, success/failure status
- **Error information**: Error types and messages (no personal data or slide content)
- **Configuration**: Segmentation config used, cancer subtype settings (no patient data)
- **HF user info** (Spaces only): HuggingFace username and login status for logged-in users

### What is NOT Collected

- **No slide content**: Images, pixel data, or pathology results are never uploaded
- **No patient data**: No PHI, patient identifiers, or clinical information
- **No file paths**: Local file paths or filenames are not collected
- **No authentication tokens**: API keys and credentials are never logged

### How to Opt Out

Telemetry is **only active on HuggingFace Spaces** and can be disabled:

1. **Environment Variable** (recommended):
   ```bash
   export MOSAIC_TELEMETRY_ENABLED=false
   ```

2. **Local installations**: Telemetry is automatically disabled for local/Docker deployments

### Data Storage

- Telemetry data is stored in a private HuggingFace dataset (if configured)
- Data is used only for improving Mosaic's performance and user experience
- No telemetry data is shared with third parties

### Telemetry Reports

A reporting script is included to generate usage summaries from collected telemetry data:

```bash
# Full report (all time)
python scripts/telemetry_report.py /path/to/telemetry

# Daily report for yesterday
python scripts/telemetry_report.py /path/to/telemetry --daily

# Report for a specific date
python scripts/telemetry_report.py /path/to/telemetry --date 2026-01-20

# Pull data from HuggingFace Dataset and generate report
python scripts/telemetry_report.py --hf-repo PDM-Group/mosaic-telemetry

# HTML format for email
python scripts/telemetry_report.py /path/to/telemetry --format html

# Email report (skip if no data)
python scripts/telemetry_report.py /path/to/telemetry --daily --email team@example.com --skip-empty
```

Reports include the following sections:

- **Cost Summary**: App uptime, active vs idle time, estimated cost at the configured hourly rate
- **Usage Summary**: Analysis counts, slides processed, breakdowns by site type and segmentation config
- **User Summary**: Logged-in vs anonymous user counts, per-user analysis and slide totals
- **Resource Summary**: Total processing time, tile counts, peak GPU memory
- **Failures**: Error type counts and recent failure messages

Example output:

```
============================================================
MOSAIC TELEMETRY REPORT for 2026-02-05
============================================================
Generated: 2026-02-06T12:00:00Z

=== COST SUMMARY ===
App sessions: 1
Total uptime: 10.00 hours
  - Active analysis: 0.26 hrs (2.6%)
  - Idle time: 9.74 hrs (97.4%)
Estimated cost: $4.00 (@ $0.4/hr)
Cost per analysis: $1.00

=== USAGE SUMMARY ===
Analyses started: 4
Analyses completed: 4
Successful analyses: 4
Total slides processed: 11
Unique sessions: 4
Average analysis duration: 231.4s

=== USER SUMMARY ===
Logged-in users: 3
Anonymous sessions: 1

By user:
  dr_smith: 2 analyses, 5 slides
  onc_research_lab: 1 analyses, 5 slides

=== RESOURCE SUMMARY ===
Total slide processing time: 0.26 hours
Total tiles processed: 68,055
Peak GPU memory: 14.10 GB

=== NO FAILURES ===

============================================================
```

For automated daily reports, add a cron entry:

```
0 8 * * * python /app/scripts/telemetry_report.py /data/telemetry --daily --email team@example.com --skip-empty
```

### Transparency

- Full telemetry implementation is in `src/mosaic/telemetry/`
- Review `src/mosaic/telemetry/events.py` to see exactly what is logged
- All telemetry code is open source and auditable

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Architecture

For detailed information about the code structure and module organization, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

This project is licensed under the terms specified in the LICENSE file.
