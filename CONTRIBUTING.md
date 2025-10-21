# Contributing to Mosaic

Thank you for your interest in contributing to Mosaic! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with CUDA support (for model inference)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/pathology-data-mining/mosaic.git
cd mosaic
```

2. Install dependencies including development tools:

```bash
uv sync
```

This will install all dependencies, including development tools like pytest, pylint, and black.

### Running Tests

Run all tests:

```bash
pytest tests/
```

Run tests with coverage report:

```bash
pytest tests/ --cov=src/mosaic --cov-report=term-missing
```

Run a specific test file:

```bash
pytest tests/inference/test_data.py -v
```

### Code Quality

#### Linting

We use pylint for code linting. Run it with:

```bash
pylint src/mosaic
```

#### Code Formatting

We use black for code formatting. Format your code with:

```bash
black src/mosaic tests/
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions, classes, and modules
- Keep functions focused and concise
- Use type hints where appropriate

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this error is raised
    """
    pass
```

### Commit Messages

- Use clear and descriptive commit messages
- Start with a verb in the imperative mood (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Provide additional context in the commit body if needed

Example:

```
Add docstrings to inference module functions

- Added comprehensive docstrings to all public functions
- Included type hints for better code clarity
- Updated existing docstrings to follow Google style
```

## Testing

### Writing Tests

- Write tests for all new features and bug fixes
- Place tests in the appropriate directory under `tests/`
- Use pytest fixtures for common setup code
- Mock external dependencies (e.g., model loading, network requests)
- Ensure tests can run without GPU access or large model downloads

### Test Structure

```python
import pytest
from mosaic.module import function_to_test

def test_function_basic_case():
    """Test basic functionality of the function."""
    result = function_to_test(input_data)
    assert result == expected_output

def test_function_edge_case():
    """Test edge cases."""
    with pytest.raises(ValueError):
        function_to_test(invalid_input)
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, focused commits
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure code quality**:
   ```bash
   black src/mosaic tests/
   pylint src/mosaic
   pytest tests/
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**:
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Provide a clear description of your changes
   - Reference any related issues

### Pull Request Guidelines

- Keep pull requests focused on a single feature or fix
- Update documentation for any changed functionality
- Add or update tests as appropriate
- Ensure all tests pass before submitting
- Respond to review feedback promptly

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, GPU model)
- Relevant log output or error messages
- Minimal code example to reproduce the issue

### Feature Requests

When suggesting a feature, please include:

- A clear description of the feature
- The use case and benefits
- Any alternative solutions you've considered
- Examples of how the feature would be used

### Issue Templates

Please use the appropriate issue template when creating a new issue.

## Development Guidelines

### Module Organization

- Keep modules focused on a single responsibility
- Place UI-related code in `src/mosaic/ui/`
- Place inference code in `src/mosaic/inference/`
- Place analysis logic in `src/mosaic/analysis.py`
- Avoid circular dependencies

### Adding New Features

When adding new features:

1. Discuss the feature in an issue first
2. Follow the existing code structure
3. Add comprehensive tests
4. Update relevant documentation
5. Consider backward compatibility

### Dependencies

- Avoid adding new dependencies unless necessary
- Discuss new dependencies in an issue or pull request
- Ensure dependencies are compatible with the project's license
- Pin dependency versions in `pyproject.toml`

## Questions?

If you have questions about contributing, please:

- Check existing issues and pull requests
- Open a new issue with your question
- Join our community discussions (if available)

Thank you for contributing to Mosaic!
