# Hashi Graph Model Evaluation

Since the notebook format was getting corrupted, here's a working evaluation script and instructions.

## Quick Start

Run the evaluation script:

```bash
cd /Users/andrewmccornack/code/hashi-graph
python3 notebooks/evaluation_script.py
```

## What the Evaluation Does

### 🔧 **Model Loading & Inference**
- Loads saved PyTorch models automatically
- Runs inference on val/test datasets
- Collects detailed predictions with confidence scores

### 📊 **Error Analysis & Classification**
- Classifies puzzles into error types: bridge count errors, capacity violations
- Tracks capacity utilization and constraint violations
- Analyzes error patterns by puzzle characteristics

### 🎨 **Visualization Suite**
- **Error Pattern Plots**: Histograms, bar charts, and trend analysis
- **Capacity Analysis**: Errors by node capacity with detailed breakdowns
- **Performance Breakdowns**: Accuracy by puzzle size, difficulty, and node capacity

### 📈 **Comprehensive Reporting**
- Model comparison summaries with performance metrics
- Export functionality for results and plots
- Statistical analysis across puzzle sizes, difficulties, and node capacities

## Configuration

Edit the paths in `evaluation_script.py`:

```python
MODEL_PATH = "../models/model_20241220_120000/best_model.pt"  # Path to saved model
CONFIG_PATH = "../models/model_20241220_120000/config.yaml"  # Config saved with model
DATA_ROOT = "../data"  # Path to dataset root
SPLIT = "val"  # 'val' or 'test'
LIMIT = 10  # Limit number of puzzles (None for all)
```

**Note**: The config file is now automatically saved alongside the model during training, so you can use the config from the model directory.

## Features Implemented

✅ **Model Loading**: Supports GCN, GAT, GINE, and Transformer models
✅ **Inference Pipeline**: Detailed prediction collection
✅ **Error Classification**: Bridge count errors, capacity violations
✅ **Visualization**: Error patterns, capacity analysis, confidence distributions
✅ **Export**: CSV results and PNG plots
✅ **Comprehensive Metrics**: Accuracy, perfect puzzle rate, error breakdowns

## Output Files

When you run the script, it creates:
- `../plots/error_patterns.png` - Error analysis visualizations
- `../plots/evaluation_results.csv` - Detailed results per edge
- Console output with summary statistics

## Troubleshooting

If you get import errors, make sure you're in the project root and have activated your virtual environment:

```bash
cd /Users/andrewmccornack/code/hashi-graph
source venv/bin/activate  # or however you activate your env
python3 notebooks/evaluation_script.py
```

The script will tell you exactly what went wrong and provide helpful error messages.
