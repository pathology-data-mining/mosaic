# Mosaic Architecture

## Module Structure

The Mosaic application has been refactored for better readability and maintainability. The codebase is now organized into the following modules:

### Core Modules

#### `mosaic.gradio_app` (Main Entry Point)
- **Location**: `src/mosaic/gradio_app.py`
- **Purpose**: CLI entry point and command-line argument parsing
- **Responsibilities**:
  - Command-line argument parsing
  - Model downloading and initialization
  - Single slide and batch processing CLI modes
  - Launching the Gradio web UI

#### `mosaic.analysis`
- **Location**: `src/mosaic/analysis.py`
- **Purpose**: Core slide analysis logic
- **Responsibilities**:
  - Tissue segmentation
  - Feature extraction (CTransPath and Optimus)
  - Feature filtering with marker classifier
  - Aeon inference (cancer subtype prediction)
  - Paladin inference (biomarker prediction)
- **Key Function**: `analyze_slide()`

#### `mosaic.ui` Package
- **Location**: `src/mosaic/ui/`
- **Purpose**: Gradio web interface components
- **Submodules**:
  
  - **`ui.__init__.py`**: Exports the main `launch_gradio` function
  
  - **`ui.app`**: Gradio interface definition
    - UI layout and component definitions
    - Event handlers for user interactions
    - Multi-slide analysis workflow
    - Key Functions: `launch_gradio()`, `analyze_slides()`, `set_cancer_subtype_maps()`
  
  - **`ui.utils`**: UI utility functions
    - Settings validation
    - CSV file handling
    - OncoTree API integration
    - User session directory management
    - Key Functions: `validate_settings()`, `load_settings()`, `get_oncotree_code_name()`, `create_user_directory()`

### Inference Modules

#### `mosaic.inference`
- **Location**: `src/mosaic/inference/`
- **Purpose**: ML model inference implementations
- **Submodules**:
  - `aeon.py`: Cancer subtype inference
  - `paladin.py`: Biomarker inference
  - `data.py`: Data structures and utilities

## Code Organization Benefits

1. **Separation of Concerns**: UI, analysis, and CLI logic are now clearly separated
2. **Improved Maintainability**: Each module has a single, well-defined responsibility
3. **Better Testability**: Individual modules can be tested independently
4. **Enhanced Readability**: Reduced file sizes and clear module boundaries
5. **Reusability**: Analysis functions can be imported and used without UI dependencies

## Import Flow

```
gradio_app.main()
├── download_and_process_models()
│   ├── set_cancer_subtype_maps() [from ui.app]
│   └── get_oncotree_code_name() [from ui.utils]
├── analyze_slide() [from analysis]
│   ├── segment_tissue() [from mussel]
│   ├── get_features() [from mussel]
│   ├── filter_features() [from mussel]
│   ├── run_aeon() [from inference]
│   └── run_paladin() [from inference]
└── launch_gradio() [from ui]
    ├── analyze_slides() [from ui.app]
    │   └── analyze_slide() [from analysis]
    └── validate_settings() [from ui.utils]
```

## File Size Comparison

| File | Original | Refactored | Change |
|------|----------|------------|--------|
| `gradio_app.py` | 843 lines | 230 lines | -73% |
| UI Components | - | 474 lines | +474 |
| Analysis Logic | - | 200 lines | +200 |

The refactoring distributed the original monolithic file into focused, maintainable modules while maintaining all functionality.
