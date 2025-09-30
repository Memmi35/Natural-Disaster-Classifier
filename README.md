# IncidentVision: Multimodal Analysis of Natural Disasters through Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive platform for real-time natural disaster classification using ResNet-18 architecture and web-based deployment. This project combines deep learning, computer vision, and modern web technologies to detect and analyze natural disasters from images.

## 🌟 Features

- **Real-time Image Classification**: Classify natural disasters into 4 categories (fire, earthquake, heavy rainfall, fog)
- **High Accuracy**: Achieves ~90% accuracy with F1-score of 0.88
- **Web Dashboard**: Interactive Next.js frontend with admin and user interfaces
- **AI-Powered Analysis**: Integration with Google Gemini API for incident summaries and recommendations
- **Professional Architecture**: Modular, scalable, and maintainable codebase
- **Comprehensive Documentation**: Well-documented notebooks and code

## 🏗️ Project Structure

```
IncidentVision-Professional/
├── data/                          # Data storage
│   ├── raw/                       # Original data files
│   ├── processed/                 # Processed datasets
│   └── external/                  # External data sources
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration_incident1m.ipynb
│   ├── 02_data_validation_url_verification.ipynb
│   ├── 03_dataset_subset_creation.ipynb
│   ├── 04_pytorch_dataset_preparation.ipynb
│   ├── 05_model_training_resnet18.ipynb
│   └── 06_model_evaluation_analysis.ipynb
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   ├── features/                 # Data preprocessing
│   ├── visualization/            # Plotting utilities
│   └── api/                      # API endpoints
├── web/                          # Web application
│   ├── frontend/                 # Next.js application
│   └── backend/                  # Python API server
├── models/                       # Trained model files
├── configs/                      # Configuration files
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
└── deployments/                  # Deployment configurations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- CUDA-capable GPU (optional but recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/IncidentVision-Professional.git
   cd IncidentVision-Professional
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up the web frontend**:
   ```bash
   cd web/frontend
   npm install
   npm run dev
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

### Usage

1. **Data Exploration**:
   Start with the notebooks in order:
   ```bash
   jupyter lab notebooks/01_data_exploration_incident1m.ipynb
   ```

2. **Model Training**:
   ```bash
   python scripts/train_model.py --config configs/resnet18_config.yaml
   ```

3. **Web Application**:
   ```bash
   # Start the API server
   uvicorn src.api:app --reload --port 8000
   
   # Start the frontend (in another terminal)
   cd web/frontend && npm run dev
   ```

4. **Make Predictions**:
   ```bash
   python scripts/predict.py --image path/to/image.jpg --model models/best_model.ckpt
   ```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 90.2% |
| F1-Score | 0.88 |
| Precision | 0.89 |
| Recall | 0.87 |

### Class-wise Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Fire | 0.92 | 0.88 | 0.90 |
| Earthquake | 0.87 | 0.89 | 0.88 |
| Heavy Rainfall | 0.89 | 0.85 | 0.87 |
| Fog | 0.88 | 0.87 | 0.87 |

## 🛠️ Technical Details

### Model Architecture
- **Base Model**: ResNet-18 with ImageNet pre-training
- **Framework**: PyTorch Lightning
- **Optimizer**: Adam (lr=3e-4)
- **Scheduler**: StepLR (step_size=5, gamma=0.9)
- **Loss Function**: CrossEntropyLoss

### Dataset
- **Source**: Incident1M dataset
- **Classes**: 4 (fire, earthquake, heavy_rainfall, fog)
- **Size**: 935 images
- **Split**: 70% train, 20% validation, 10% test

### Web Technologies
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python
- **AI Integration**: Google Gemini API
- **UI Components**: Radix UI, shadcn/ui

## 🔧 Configuration

### Environment Variables

```env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key
MODEL_PATH=models/best_model.ckpt
LABEL_MAPPING_PATH=configs/label_mapping.json

# Database (if applicable)
DATABASE_URL=sqlite:///./incidents.db

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Model Configuration (`configs/resnet18_config.yaml`)

```yaml
model:
  architecture: resnet18
  num_classes: 4
  pretrained: true

training:
  batch_size: 32
  learning_rate: 0.0003
  max_epochs: 15
  patience: 3

data:
  train_csv: data/processed/train_f.csv
  val_csv: data/processed/val_f.csv
  test_csv: data/processed/test_f.csv
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Model Training Guide](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing Guidelines](docs/contributing.md)

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run notebook tests
python scripts/test_notebooks.py
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Cloud Deployment

See [deployment guide](docs/deployment.md) for detailed instructions on deploying to:
- AWS EC2/ECS
- Google Cloud Platform
- Azure Container Instances

## 📈 Monitoring and Logging

The application includes comprehensive logging and monitoring:

- **Structured Logging**: JSON formatted logs
- **Metrics Collection**: Prometheus compatible metrics
- **Error Tracking**: Sentry integration
- **Performance Monitoring**: Request timing and resource usage

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **E.O Slimane Yassine** - *Initial work* - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- **Qatar Computing Research Institute (QCRI)** for providing access to the Incident1M dataset
- Incident1M dataset creators for their comprehensive work
- PyTorch and PyTorch Lightning teams
- Next.js and React communities
- Google Gemini API team
- Military Academy for support and resources

## 📧 Contact

- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

## 🔗 Related Work

- [Original Research Paper](link-to-paper)
- [Incident1M Dataset](https://github.com/ethanweber/incident1m)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

---

⭐ **Star this repository if you find it helpful!**