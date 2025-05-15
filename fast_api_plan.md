# FastAPI Integration Plan

## Implementation Steps

1. **Set up a basic FastAPI application structure**:

```python
# api/main.py
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import sys
import pandas as pd

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your recommendation system modules
from config import load_config
from data.loaders import load_retail_dataset
from models.base import BaseRecommender
from models.popular import PopularityRecommender
from models.content_based import ContentBasedRecommender
from models.collaborative.matrix_fact import MatrixFactorizationRecommender
from models.hybrid import HybridRecommender
from models.cold_start import ColdStartRecommender

# Create FastAPI app
app = FastAPI(
    title="Recommendation System API",
    description="API for e-commerce recommendation system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models and dataset
config = None
dataset = None
models = {}

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    user_id: int
    n: Optional[int] = 10
    exclude_seen: Optional[bool] = True
    model: Optional[str] = "hybrid"
    
class RecommendationItem(BaseModel):
    item_id: int
    score: float
    name: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    
class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    model_used: str
    
class DetailedRecommendationItem(BaseModel):
    item_id: int
    score: float
    name: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    explanation: Optional[Dict[str, Any]] = None
    
class DetailedRecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[DetailedRecommendationItem]
    model_used: str

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    global config, dataset, models
    
    # Load configuration
    config = load_config("config_base.yaml").config_data
    
    # Load dataset
    dataset = load_retail_dataset(config)
    
    # Load models
    models["popular"] = PopularityRecommender(config)
    models["content_based"] = ContentBasedRecommender(config)
    models["matrix_fact"] = MatrixFactorizationRecommender(config)
    models["hybrid"] = HybridRecommender(config)
    models["cold_start"] = ColdStartRecommender(config)
    
    # Set ID mappings for all models
    for name, model in models.items():
        model.set_id_mappings(
            dataset.get_user_id_mapping(),
            dataset.get_item_id_mapping()
        )
    
    # Train models
    train_data = dataset.get_interaction_data()
    item_features = dataset.get_item_features()
    user_features = dataset.get_user_features()
    user_segments = dataset.get_customer_segments()
    
    # Train each model
    models["popular"].fit(train_data)
    models["content_based"].fit(train_data, item_features, user_features)
    models["matrix_fact"].fit(train_data)
    
    # Set up hybrid model
    hybrid = models["hybrid"]
    hybrid.add_recommender('collaborative', models["matrix_fact"], weight=0.7)
    hybrid.add_recommender('content_based', models["content_based"], weight=0.3)
    hybrid.fit(train_data)
    
    # Train cold start model
    models["cold_start"].fit(train_data, item_features, user_features, user_segments)

# Dependency to get model by name
def get_model(model_name: str = "hybrid"):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return models[model_name]

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the Recommendation System API"}

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    model = get_model(request.model)
    
    # Get user's seen items
    seen_items = None
    if request.exclude_seen and hasattr(dataset, 'interactions'):
        user_interactions = dataset.interactions[dataset.interactions['customer_key'] == request.user_id]
        seen_items = user_interactions['product_key'].tolist() if not user_interactions.empty else []
    
    # Generate recommendations
    recs = model.recommend(
        user_id=request.user_id,
        n=request.n,
        exclude_seen=request.exclude_seen,
        seen_items=seen_items
    )
    
    # Format recommendations
    recommendation_items = []
    for item_id, score in recs:
        item = {
            "item_id": item_id,
            "score": float(score)
        }
        
        # Add item name if available
        if hasattr(dataset, 'item_features') and item_id in dataset.item_features.index:
            item_features = dataset.item_features.loc[item_id]
            if 'product_name' in item_features:
                item["name"] = item_features['product_name']
            if 'category' in item_features:
                item["category"] = item_features['category']
        
        recommendation_items.append(RecommendationItem(**item))
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendation_items,
        model_used=request.model
    )

@app.post("/recommendations/detailed", response_model=DetailedRecommendationResponse)
async def get_detailed_recommendations(request: RecommendationRequest):
    model = get_model(request.model)
    
    # Check if model supports detailed recommendations
    if not hasattr(model, 'recommend_with_details') and request.model != "cold_start":
        # Fall back to regular recommendations
        response = await get_recommendations(request)
        return DetailedRecommendationResponse(
            user_id=response.user_id,
            recommendations=[
                DetailedRecommendationItem(**item.dict())
                for item in response.recommendations
            ],
            model_used=response.model_used
        )
    
    # Get user's seen items
    seen_items = None
    if request.exclude_seen and hasattr(dataset, 'interactions'):
        user_interactions = dataset.interactions[dataset.interactions['customer_key'] == request.user_id]
        seen_items = user_interactions['product_key'].tolist() if not user_interactions.empty else []
    
    # For cold start model, we need special handling
    if request.model == "cold_start":
        # Get user interaction count
        interaction_count = 0
        if hasattr(dataset, 'interactions'):
            user_interactions = dataset.interactions[dataset.interactions['customer_key'] == request.user_id]
            interaction_count = len(user_interactions)
        
        # Determine if user is cold-start
        is_cold_start = interaction_count < model.min_interactions
        
        # Generate explanations based on segment
        segment = None
        if hasattr(dataset, 'user_segments'):
            for seg_name, user_list in dataset.user_segments.items():
                if request.user_id in user_list:
                    segment = seg_name
                    break
        
        # Generate recommendations with explanations
        recs = model.recommend(
            user_id=request.user_id,
            n=request.n,
            exclude_seen=request.exclude_seen,
            seen_items=seen_items
        )
        
        # Format recommendations
        recommendation_items = []
        for item_id, score in recs:
            item = {
                "item_id": item_id,
                "score": float(score),
                "explanation": {
                    "is_cold_start": is_cold_start,
                    "user_segment": segment,
                    "recommendation_strategy": "demographic_similarity" if is_cold_start else "content_based"
                }
            }
            
            # Add item name and category if available
            if hasattr(dataset, 'item_features') and item_id in dataset.item_features.index:
                item_features = dataset.item_features.loc[item_id]
                if 'product_name' in item_features:
                    item["name"] = item_features['product_name']
                if 'category' in item_features:
                    item["category"] = item_features['category']
                    item["explanation"]["category"] = item_features['category']
            
            recommendation_items.append(DetailedRecommendationItem(**item))
        
        return DetailedRecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendation_items,
            model_used=request.model
        )
    
    # Generate detailed recommendations
    recs = model.recommend_with_details(
        user_id=request.user_id,
        n=request.n,
        exclude_seen=request.exclude_seen,
        seen_items=seen_items
    )
    
    # Format recommendations
    recommendation_items = []
    for rec in recs:
        item_id = rec["item_id"]
        item = {
            "item_id": item_id,
            "score": float(rec["score"])
        }
        
        # Add confidence if available
        if "confidence" in rec:
            item["confidence"] = float(rec["confidence"])
        
        # Add explanation if available
        if "explanation" in rec:
            item["explanation"] = rec["explanation"]
        
        # Add item name if available
        if "item_name" in rec:
            item["name"] = rec["item_name"]
        elif hasattr(dataset, 'item_features') and item_id in dataset.item_features.index:
            item_features = dataset.item_features.loc[item_id]
            if 'product_name' in item_features:
                item["name"] = item_features['product_name']
            if 'category' in item_features:
                item["category"] = item_features['category']
        
        recommendation_items.append(DetailedRecommendationItem(**item))
    
    return DetailedRecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendation_items,
        model_used=request.model
    )

@app.get("/models")
async def get_available_models():
    return {"models": list(models.keys())}

@app.get("/users")
async def get_users(limit: int = Query(10, ge=1, le=100), 
                   segment: Optional[str] = None):
    if not hasattr(dataset, 'customers'):
        raise HTTPException(status_code=404, detail="User data not available")
    
    # Filter by segment if specified
    if segment and hasattr(dataset, 'user_segments') and segment in dataset.user_segments:
        segment_users = dataset.user_segments[segment]
        users = dataset.customers[dataset.customers['customer_key'].isin(segment_users)]
    else:
        users = dataset.customers
    
    # Return limited number of users
    user_list = users.head(limit)['customer_key'].tolist()
    
    return {"users": user_list}

@app.get("/items")
async def get_items(limit: int = Query(10, ge=1, le=100),
                   category: Optional[str] = None):
    if not hasattr(dataset, 'products'):
        raise HTTPException(status_code=404, detail="Item data not available")
    
    # Filter by category if specified
    if category and 'category' in dataset.products.columns:
        items = dataset.products[dataset.products['category'] == category]
    else:
        items = dataset.products
    
    # Return limited number of items with basic info
    item_list = []
    for _, item in items.head(limit).iterrows():
        item_dict = {
            "item_id": item['product_key']
        }
        
        # Add additional fields if available
        if 'product_name' in item:
            item_dict["name"] = item['product_name']
        if 'category' in item:
            item_dict["category"] = item['category']
        if 'cost' in item:
            item_dict["price"] = float(item['cost'])
        
        item_list.append(item_dict)
    
    return {"items": item_list}

@app.get("/user/{user_id}")
async def get_user_info(user_id: int):
    if not hasattr(dataset, 'customers'):
        raise HTTPException(status_code=404, detail="User data not available")
    
    # Check if user exists
    if user_id not in dataset.customers['customer_key'].values:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    # Get user data
    user_data = dataset.customers[dataset.customers['customer_key'] == user_id].iloc[0].to_dict()
    
    # Get user segment if available
    user_segment = None
    if hasattr(dataset, 'user_segments'):
        for segment, users in dataset.user_segments.items():
            if user_id in users:
                user_segment = segment
                break
    
    # Get interaction count
    interaction_count = 0
    if hasattr(dataset, 'interactions'):
        user_interactions = dataset.interactions[dataset.interactions['customer_key'] == user_id]
        interaction_count = len(user_interactions)
    
    # Create response
    response = {
        "user_id": user_id,
        "user_data": user_data,
        "segment": user_segment,
        "interaction_count": interaction_count
    }
    
    return response

# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
```

2. **Create a simple configuration file for the API**:

```python
# api/settings.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    API_TITLE: str = "Recommendation System API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for e-commerce recommendation system"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    CONFIG_FILE: str = "config_base.yaml"
    
    # Enable/disable features
    ENABLE_EXPLANATIONS: bool = True
    ENABLE_CONFIDENCE: bool = True
    ENABLE_TIME_DECAY: bool = True
    
    class Config:
        env_file = ".env"
        
settings = Settings()
```

3. **Add requirements for API to requirements.txt**:

```
# API dependencies
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0
python-dotenv>=0.19.0
```

4. **Create a script to run the API**:

```python
# run_api.py
import uvicorn
from api.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )
```

5. **Create a simple test client for the API**:

```python
# api/test_client.py
import requests
import json
from typing import Dict, List, Any

class RecommendationClient:
    """
    Client for interacting with the recommendation system API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        
    def get_recommendations(self, user_id: int, n: int = 10, 
                          exclude_seen: bool = True, 
                          model: str = "hybrid",
                          detailed: bool = False) -> Dict[str, Any]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            model: Recommendation model to use
            detailed: Whether to get detailed recommendations
            
        Returns:
            Recommendation response
        """
        # Prepare request
        endpoint = "/recommendations/detailed" if detailed else "/recommendations"
        url = f"{self.base_url}{endpoint}"
        payload = {
            "user_id": user_id,
            "n": n,
            "exclude_seen": exclude_seen,
            "model": model
        }
        
        # Make request
        response = requests.post(url, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
        # Parse response
        return response.json()
        
    def get_available_models(self) -> List[str]:
        """
        Get list of available recommendation models.
        
        Returns:
            List of model names
        """
        response = requests.get(f"{self.base_url}/models")
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return []
            
        return response.json()["models"]
        
    def get_users(self, limit: int = 10, segment: str = None) -> List[int]:
        """
        Get list of users.
        
        Args:
            limit: Maximum number of users to return
            segment: Optional segment filter
            
        Returns:
            List of user IDs
        """
        params = {"limit": limit}
        if segment:
            params["segment"] = segment
            
        response = requests.get(f"{self.base_url}/users", params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return []
            
        return response.json()["users"]
        
    def get_items(self, limit: int = 10, category: str = None) -> List[Dict[str, Any]]:
        """
        Get list of items.
        
        Args:
            limit: Maximum number of items to return
            category: Optional category filter
            
        Returns:
            List of item dictionaries
        """
        params = {"limit": limit}
        if category:
            params["category"] = category
            
        response = requests.get(f"{self.base_url}/items", params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return []
            
        return response.json()["items"]
        
    def get_user_info(self, user_id: int) -> Dict[str, Any]:
        """
        Get information about a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User information dictionary
        """
        response = requests.get(f"{self.base_url}/user/{user_id}")
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return None
            
        return response.json()

# Example usage
if __name__ == "__main__":
    client = RecommendationClient()
    
    # Get available models
    models = client.get_available_models()
    print(f"Available models: {models}")
    
    # Get some users
    users = client.get_users(limit=5)
    if users:
        print(f"Sample users: {users}")
        
        # Get recommendations for first user
        user_id = users[0]
        print(f"\nGetting recommendations for user {user_id}...")
        
        # Try different models
        for model in models:
            print(f"\nUsing model: {model}")
            recs = client.get_recommendations(user_id, n=5, model=model)
            
            if recs:
                print(f"Recommendations:")
                for i, item in enumerate(recs["recommendations"]):
                    name = item.get("name", f"Item {item['item_id']}")
                    print(f"  {i+1}. {name} (Score: {item['score']:.4f})")
                    
        # Get detailed recommendations
        print("\nGetting detailed recommendations...")
        detailed_recs = client.get_recommendations(user_id, n=3, model="hybrid", detailed=True)
        
        if detailed_recs:
            print(f"Detailed Recommendations:")
            for i, item in enumerate(detailed_recs["recommendations"]):
                name = item.get("name", f"Item {item['item_id']}")
                confidence = item.get("confidence", "N/A")
                print(f"  {i+1}. {name} (Score: {item['score']:.4f}, Confidence: {confidence})")
                
                if "explanation" in item:
                    print(f"     Explanation: {json.dumps(item['explanation'], indent=2)}")
```

## API Endpoints Overview

The FastAPI application provides the following endpoints:

1. **GET /** - Welcome message
2. **POST /recommendations** - Get basic recommendations for a user
3. **POST /recommendations/detailed** - Get detailed recommendations with explanations
4. **GET /models** - List available recommendation models
5. **GET /users** - List sample users (with optional segment filter)
6. **GET /items** - List sample items (with optional category filter)
7. **GET /user/{user_id}** - Get information about a specific user

## Integration Steps

1. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn pydantic python-dotenv
   ```

2. **Create API Directory Structure**:
   ```
   api/
   ├── __init__.py
   ├── main.py
   ├── settings.py
   └── test_client.py
   run_api.py
   ```

3. **Start the API**:
   ```bash
   python run_api.py
   ```

4. **Access the Swagger Documentation**:
   Open a browser and navigate to `http://localhost:8000/docs`

5. **Test the API with the Client**:
   ```bash
   python api/test_client.py
   ```

## Deployment Considerations

For production deployment, consider:

1. **Docker Containerization**:
   Create a Dockerfile for easy deployment:

   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   CMD ["python", "run_api.py"]
   ```

2. **Environment Configuration**:
   Create a `.env` file for environment-specific settings:

   ```
   HOST=0.0.0.0
   PORT=8000
   CONFIG_FILE=config_prod.yaml
   ENABLE_EXPLANATIONS=true
   ```

3. **Securing the API**:
   Add authentication using FastAPI's security utilities:

   ```python
   from fastapi.security import APIKeyHeader
   from fastapi import Security, HTTPException, status

   API_KEY = "your_secret_key"
   api_key_header = APIKeyHeader(name="X-API-Key")

   def get_api_key(api_key: str = Security(api_key_header)):
       if api_key != API_KEY:
           raise HTTPException(
               status_code=status.HTTP_401_UNAUTHORIZED,
               detail="Invalid API Key"
           )
       return api_key
   ```

   Then apply it to endpoints:

   ```python
   @app.post("/recommendations", response_model=RecommendationResponse)
   async def get_recommendations(request: RecommendationRequest, api_key: str = Depends(get_api_key)):
       # ...
   ```

4. **Rate Limiting**:
   Add rate limiting to prevent abuse:

   ```python
   from fastapi.middleware import Middleware
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   from slowapi.middleware import SlowAPIMiddleware

   limiter = Limiter(key_func=get_remote_address)
   app = FastAPI(middleware=[Middleware(SlowAPIMiddleware)])
   app.state.limiter = limiter

   @app.post("/recommendations")
   @limiter.limit("10/minute")
   async def get_recommendations(request: RecommendationRequest):
       # ...
   ```