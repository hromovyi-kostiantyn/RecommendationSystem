# E-Commerce Recommendation System

A comprehensive recommendation system for e-commerce applications, supporting multiple recommendation strategies, cold-start handling, and explanations.

## Features

- 📊 **Multiple recommendation algorithms**:
  - Content-based recommendations
  - Collaborative filtering (Neighborhood and Matrix Factorization)
  - Popularity-based recommendations
  - Hybrid recommendations
  - Cold-start recommendations

- 🕰️ **Time-aware recommendations**:
  - Time decay for recent interactions
  - Session-based recommendations
  - Sequential patterns

- 🔍 **Explainable recommendations**:
  - Recommendation explanations
  - Confidence scores
  - User segment analysis

- 🚀 **Performance optimizations**:
  - Memory-efficient sparse matrix operations
  - Incremental learning
  - Batch processing

- 🌐 **API Integration**:
  - FastAPI REST API
  - Interactive Swagger documentation
  - Client library

## System Architecture

![System Architecture](docs/system_architecture.png)

The system consists of the following main components:

1. **Data Layer**: Handles data loading, preprocessing, and feature extraction
2. **Model Layer**: Provides various recommendation algorithms
3. **Evaluation Layer**: Metrics and tools for evaluating recommendation quality
4. **API Layer**: Exposes recommendations via REST API

## Recommendation Models

| Model | Description | Best For |
|-------|-------------|----------|
| `PopularityRecommender` | Recommends most popular items | New users with no history |
| `ContentBasedRecommender` | Recommends based on item features | Users with clear preferences |
| `NeighborhoodRecommender` | User-based or item-based collaborative filtering | Established users with history |
| `MatrixFactorizationRecommender` | Matrix factorization using SVD | General purpose recommendations |
| `ColdStartRecommender` | Special handling for new users | New users with demographic info |
| `HybridRecommender` | Combines multiple recommendation approaches | Best overall performance |
| `SessionBasedRecommender` | Sequential recommendation patterns | Capturing short-term interests |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Getting Started

### Data Preparation

1. Place your dataset files in the `data/datasets` directory:
   - `g_dim_customers.csv`: Customer information
   - `g_dim_products.csv`: Product information
   - `g_fact_sales.csv`: Sales/interaction data
   - `Customer_report_cleaned_data.csv`: Additional customer features
   - `Product_report_cleaned_data.csv`: Additional product features

2. Configure dataset settings in `config/config_base.yaml`

### Training and Evaluation

```bash
# Train and evaluate all models
python main.py --mode evaluate --models all

# Train and evaluate a specific model
python main.py --mode evaluate --models hybrid

# Train models only
python main.py --mode train --models content_based hybrid
```

### Recommendation Generation

```bash
# Generate recommendations for a specific user
python main.py --mode recommend --models hybrid --user_id 1234 --n_recommendations 10
```

### API Service

```bash
# Start the API server
python run_api.py

# Access the API documentation
# Open http://localhost:8000/docs in your browser
```

## Configuration

The system is configured using YAML files in the `config` directory:

- `config_base.yaml`: Base configuration
- `config_experiments/`: Experiment-specific configurations

Main configuration options:

```yaml
# Data settings
data:
  sample_size: 1.0  # Use 100% of data
  random_seed: 42

# Splitting strategy for evaluation
split:
  test_size: 0.2
  strategy: "random"  # Options: random, time, user, item
  min_ratings: 5

# Model parameters
models:
  content_based:
    # ...
  collaborative:
    # ...
  hybrid:
    weights:
      content_based: 0.3
      collaborative: 0.7
  cold_start:
    min_interactions: 5
    # ...

# Time-aware settings
time_aware:
  half_life_days: 30.0
  # ...

# Evaluation settings
evaluation:
  metrics: ["precision", "recall", "ndcg", "hit_rate", "map", "mrr"]
  k_values: [1, 5, 10, 20]
```

## Performance Metrics

Our system achieved the following performance on the retail dataset:

| Model | MRR | NDCG@10 | Hit Rate@10 | Recall@10 |
|-------|-----|---------|-------------|-----------|
| Random | 0.0412 | 0.0419 | 0.0238 | 0.0216 |
| Popularity | 0.0448 | 0.2125 | 0.0193 | 0.0177 |
| Content-Based | 0.1640 | 0.1708 | 0.0846 | 0.0778 |
| Neighborhood | 0.0847 | 0.0211 | 0.0521 | 0.0454 |
| Matrix Factorization | 0.1134 | 0.0274 | 0.0595 | 0.0521 |
| Hybrid | 0.2618 | 0.0535 | 0.0796 | 0.0716 |
| Cold Start | 0.1640 | 0.1708 | 0.0846 | 0.0778 |

## User Segments Analysis

Performance varies across user segments:

| Segment | Model | Hit Rate@10 | Recall@10 |
|---------|-------|-------------|-----------|
| VIP | Hybrid | 0.1856 | 0.1599 |
| Regular | Hybrid | 0.0993 | 0.0981 |
| New | Hybrid | 0.0125 | 0.0125 |
| VIP | Cold Start | 0.1572 | 0.1382 |
| Regular | Cold Start | 0.1114 | 0.1090 |
| New | Cold Start | 0.0342 | 0.0331 |

## API Usage

```python
import requests

# Get recommendations
response = requests.post(
    "http://localhost:8000/recommendations",
    json={
        "user_id": 1234,
        "n": 5,
        "exclude_seen": True,
        "model": "hybrid"
    }
)

recommendations = response.json()["recommendations"]
for i, item in enumerate(recommendations):
    print(f"{i+1}. {item['name']} (Score: {item['score']:.4f})")

# Get detailed recommendations with explanations
response = requests.post(
    "http://localhost:8000/recommendations/detailed",
    json={
        "user_id": 1234,
        "model": "cold_start"
    }
)

detailed_recs = response.json()["recommendations"]
for i, item in enumerate(detailed_recs):
    print(f"{i+1}. {item['name']} (Score: {item['score']:.4f})")
    print(f"   Explanation: {item['explanation']}")
```

## Project Structure

```
recommendation-system/
├── api/                    # API layer
├── config/                 # Configuration files
├── data/                   # Data processing modules
│   ├── datasets/           # Dataset files
│   ├── dataset.py          # Dataset class
│   ├── loaders.py          # Data loading utilities
│   ├── preprocessing.py    # Data preprocessing
│   └── splitters.py        # Train/test splitting strategies
├── evaluation/             # Evaluation modules
│   ├── evaluator.py        # Evaluation framework
│   └── metrics.py          # Evaluation metrics
├── experiments/            # Experiment utilities
│   ├── analyzer.py         # Results analysis
│   └── runner.py           # Experiment runner
├── models/                 # Recommendation models
│   ├── collaborative/      # Collaborative filtering models
│   ├── base.py             # Base recommender interface
│   ├── cold_start.py       # Cold-start handling
│   ├── content_based.py    # Content-based recommender
│   ├── explainability.py   # Recommendation explanations
│   ├── hybrid.py           # Hybrid recommender
│   ├── incremental.py      # Incremental learning
│   ├── popular.py          # Popularity-based recommender
│   ├── random_recommender.py # Random baseline
│   ├── sequential.py       # Sequential recommender
│   └── time_aware.py       # Time-aware functionality
├── utils/                  # Utility functions
│   ├── helpers.py          # Helper utilities
│   ├── logger.py           # Logging configuration
│   ├── results.py          # Results processing
│   ├── sparse_utils.py     # Sparse matrix utilities
│   ├── time_utils.py       # Time-related utilities
│   └── visualization.py    # Visualization utilities
├── main.py                 # Main application entry point
├── run_api.py              # API server entry point
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Future Improvements

- Add support for real-time recommendation updates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped with the development of this project
- Special thanks to the open-source community for providing the tools and libraries that made this project possible