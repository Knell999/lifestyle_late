# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
- **Web Interface (Recommended)**: `./run_streamlit_uv.sh` - Runs Streamlit app using UV package manager
- **Web Interface (Standard)**: `streamlit run streamlit_app.py` - Runs with standard Python environment
- **CLI Interface**: `./run.sh` or `python main.py` - Command line ML pipeline

### Development Tools
- **Project Verification**: `./verify_project.sh` - Comprehensive health check of all components
- **Test Modules**: `python test_modules.py` - Basic module import testing

### Package Management
- **Primary**: Uses UV (modern Python package manager) for dependency management
- **Fallback**: Standard pip with requirements.txt
- **Install Dependencies**: `uv pip install -r requirements.txt` or `pip install -r requirements.txt`

## Project Architecture

### Core Structure
This is a machine learning project for KCB credit grade prediction with both web and CLI interfaces:

```
src/                    # Core ML modules
├── config.py          # Centralized configuration (feature definitions, model params)
├── pipeline.py        # Main ML pipelines (MLPipeline, ModeComparisonPipeline)
├── preprocessing/     # Data preprocessing and feature engineering
├── models/           # Model training (Random Forest, XGBoost, LightGBM, Ensemble)
├── evaluation/       # Model evaluation and metrics
├── visualization/    # Plotting and charts
└── utils/           # Experiment tracking and helpers

ui/                   # Streamlit UI components
├── data_upload.py   # File upload and data validation
├── model_selection.py # Model training interface
├── results_display.py # Results and metrics display
└── visualization.py  # Interactive charts

streamlit_app.py     # Main Streamlit application entry point
main.py             # CLI application entry point
```

### Key Components

#### Data Processing Pipeline
- **DataLoader**: Handles CSV data loading from `data/df_KCB_grade.csv`
- **DataPreprocessor**: Feature engineering with three modes:
  - `life`: Lifestyle features only
  - `fin`: Financial features only  
  - `full`: All features combined
- **Feature Types**: OneHot, Binary, Continuous, and Financial feature groups defined in config.py

#### Model Training
- **Supported Models**: Random Forest, XGBoost, LightGBM, Logistic Regression, Ensemble
- **Training Modes**: Single model or multi-model pipeline
- **Hyperparameter Tuning**: Grid search with parameters defined in `config.py`

#### Experiment Management
- **ExperimentTracker**: Automatic saving of models, results, and configurations
- **Experiment Storage**: Results saved to `experiments/exp_YYYYMMDD_HHMMSS/`
- **Model Persistence**: Trained models saved as `.pkl` files

### Application Interfaces

#### Streamlit Web App
- Multi-page application with sidebar navigation
- Pages: Home, Data Upload, Model Training, Prediction, Visualization, Model Comparison
- Real-time progress tracking for model training
- Interactive visualizations with Plotly/Altair
- Korean language support with proper font handling

#### CLI Interface
- Interactive menu system for analysis selection
- Options: Full analysis, Mode comparison, or Both
- Automated pipeline execution with progress logging
- Results automatically saved to experiments directory

## Development Notes

### Data Requirements
- Primary dataset: `data/df_KCB_grade.csv` (managed with Git LFS)
- Target variable: `KCB_grade` (credit grade classification)
- Features include demographic, lifestyle, and financial variables

### Configuration Management
- All feature definitions and model parameters centralized in `src/config.py`
- Easy to modify feature groups and hyperparameters
- Korean feature name mappings for visualization

### Error Handling
- Comprehensive logging throughout the application
- Graceful degradation when data files are missing
- User-friendly error messages in both interfaces

### Performance Considerations
- Cross-validation with 5-fold stratified splits
- Overfitting detection through train/test performance comparison
- Ensemble methods for improved stability and performance

## Environment Setup Notes

- **Python Version**: Requires Python 3.13+ (specified in pyproject.toml)
- **Primary Dependencies**: pandas, scikit-learn, xgboost, lightgbm, streamlit
- **Visualization**: matplotlib, seaborn, plotly, altair with Korean font support
- **Package Manager**: UV recommended for faster dependency resolution