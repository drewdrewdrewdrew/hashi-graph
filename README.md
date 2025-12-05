# Python Environment Management in Cursor IDE (macOS)

## Initial Setup

1. Install `uv` (faster alternative to pip/venv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a new project directory:
   ```bash
   mkdir my_project
   cd my_project
   ```

## Environment Creation & Management

1. Create a new virtual environment:
   ```bash
   uv venv
   ```
   This creates a `.venv` directory in your project

2. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

3. Install basic development packages:
   ```bash
   uv pip install ipykernel jupyter numpy pandas python-dotenv
   ```

## Cursor IDE Configuration

1. Select Python Interpreter:
   - Cmd + Shift + P
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `.venv/bin/python`

2. Setup Jupyter Integration:
   ```bash
   python -m ipykernel install --user --name=my_project_env
   ```

## Working with Requirements

1. Create requirements.txt:
   ```bash
   uv pip freeze > requirements.txt
   ```

2. Install from requirements.txt:
   ```bash
   uv pip install -r requirements.txt
   ```

## Best Practices

1. Always activate your environment before working:
   ```bash
   source .venv/bin/activate
   ```

2. Create a `.gitignore` file:
   ```
   .venv/
   __pycache__/
   .env
   .DS_Store
   ```

3. Deactivate when switching projects:
   ```bash
   deactivate
   ```

## Jupyter Notebook Tips

1. Ensure your kernel is visible:
   - Open a notebook
   - Click kernel selector (top-right)
   - Select `my_project_env`

2. If kernel not visible:
   ```bash
   python -m ipykernel install --user --name=my_project_env
   ```

## Troubleshooting

1. If kernel fails to start:
   ```bash
   jupyter kernelspec list  # List available kernels
   jupyter kernelspec remove my_project_env  # Remove if needed
   python -m ipykernel install --user --name=my_project_env  # Reinstall
   ```

2. Environment not activating:
   - Ensure you're in project directory
   - Verify `.venv` exists
   - Try: `source .venv/bin/activate`

3. Package conflicts:
   ```bash
   uv pip list  # Check installed packages
   uv pip install package_name --upgrade  # Upgrade specific package
   ```

4. Conflicting package versions:
   ```bash
   # Check package dependencies
   uv pip show package_name  # View package requirements
   
   # Create requirements.txt with exact versions
   uv pip freeze > requirements.txt
   
   # Force reinstall problematic package
   uv pip uninstall package_name
   uv pip install package_name==specific_version
   ```

5. Common ML/Data Science package conflicts:
   - tensorflow vs pytorch (GPU versions)
   - pandas vs numpy version mismatches 
   - scikit-learn version compatibility
   - Solution: Use separate environments for different ML frameworks

6. Memory issues with large datasets:
   ```bash
   # Check memory usage in notebook
   import psutil
   print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
   
   # Clear memory
   import gc
   gc.collect()
   ```

7. GPU troubleshooting:
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # View GPU memory usage
   nvidia-smi
   ```

8. Jupyter performance tips:
   - Restart kernel regularly
   - Use `%%capture` for noisy output cells
   - `%store` for sharing variables between notebooks
   - `%load_ext autoreload` and `%autoreload 2` for auto-reloading modules
