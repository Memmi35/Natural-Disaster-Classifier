"""
API utilities for serving the IncidentVision model.

This module provides FastAPI endpoints for model inference and integrates
with the Google Gemini API for generating incident descriptions and recommendations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import numpy as np
from typing import Dict, Any
import logging
import json
import google.generativeai as genai
from datetime import datetime

from ..models import IncidentClassifier
from ..features import get_transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IncidentAPI:
    """
    Main API class for incident classification and analysis.
    """
    
    def __init__(self, model_path: str, label_mapping_path: str, gemini_api_key: str):
        """
        Initialize the API with model and configurations.
        
        Args:
            model_path: Path to the trained PyTorch model
            label_mapping_path: Path to the label mapping JSON file
            gemini_api_key: Google Gemini API key
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = IncidentClassifier.load_from_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Setup transforms
        self.transform = get_transforms('test')
        
        # Setup Gemini API
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Incident-specific recommendations
        self.recommendations = {
            'on_fire': [
                "Evacuate immediately if in danger",
                "Cover nose and mouth with cloth",
                "Stay low to avoid smoke inhalation",
                "Call emergency services (fire department)",
                "Do not use elevators",
                "Follow evacuation routes"
            ],
            'earthquake': [
                "Drop, Cover, and Hold On",
                "Stay away from windows and heavy objects",
                "If outdoors, move away from buildings",
                "Do not run outside during shaking",
                "After shaking stops, check for injuries",
                "Be prepared for aftershocks"
            ],
            'heavy_rainfall': [
                "Avoid flooded roads and areas",
                "Stay indoors if possible",
                "Keep emergency supplies ready",
                "Monitor weather updates",
                "Avoid walking in moving water",
                "Keep phones charged for emergencies"
            ],
            'fog': [
                "Reduce driving speed significantly",
                "Use low beam headlights",
                "Keep safe following distance",
                "Use fog lights if available",
                "Avoid overtaking other vehicles",
                "Pull over safely if visibility is too poor"
            ]
        }
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess uploaded image for model inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Predict incident type from image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            # Get predicted class name
            predicted_class = self.idx_to_label[predicted_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_probabilities': {
                    self.idx_to_label[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def generate_incident_summary(self, predicted_class: str, confidence: float) -> str:
        """Generate incident summary using Gemini API."""
        try:
            prompt = f"""
            An AI model has detected a {predicted_class} incident with {confidence:.1%} confidence.
            Please provide a brief, clear summary of what this incident typically involves and its characteristics.
            Keep the response under 100 words and focus on factual information.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"A {predicted_class} incident has been detected with {confidence:.1%} confidence."
    
    def get_risk_assessment(self, predicted_class: str, confidence: float) -> str:
        """Assess risk level based on incident type and confidence."""
        risk_levels = {
            'on_fire': 'High Risk',
            'earthquake': 'High Risk', 
            'heavy_rainfall': 'Moderate Risk',
            'fog': 'Low to Moderate Risk'
        }
        
        base_risk = risk_levels.get(predicted_class, 'Unknown Risk')
        
        if confidence < 0.7:
            return f"{base_risk} (Low Confidence)"
        elif confidence < 0.9:
            return f"{base_risk} (Moderate Confidence)"
        else:
            return f"{base_risk} (High Confidence)"
    
    def get_recommendations(self, predicted_class: str) -> list:
        """Get safety recommendations for the predicted incident type."""
        return self.recommendations.get(predicted_class, [
            "Stay alert and monitor the situation",
            "Follow local emergency protocols",
            "Contact emergency services if needed"
        ])
    
    def analyze_incident(self, image: Image.Image) -> Dict[str, Any]:
        """
        Complete incident analysis including prediction, summary, and recommendations.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with complete analysis results
        """
        # Get prediction
        prediction_results = self.predict(image)
        predicted_class = prediction_results['predicted_class']
        confidence = prediction_results['confidence']
        
        # Generate summary
        summary = self.generate_incident_summary(predicted_class, confidence)
        
        # Get risk assessment
        risk_assessment = self.get_risk_assessment(predicted_class, confidence)
        
        # Get recommendations
        recommendations = self.get_recommendations(predicted_class)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'summary': summary,
            'risk_assessment': risk_assessment,
            'recommended_actions': recommendations,
            'all_probabilities': prediction_results['all_probabilities']
        }


# FastAPI application
app = FastAPI(
    title="IncidentVision API",
    description="Real-time natural disaster classification and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API instance (to be initialized)
api_instance = None


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    global api_instance
    # These paths should be configured via environment variables in production
    model_path = "models/best_model.ckpt"
    label_mapping_path = "configs/label_mapping.json"
    gemini_api_key = "your_gemini_api_key"  # Set via environment variable
    
    api_instance = IncidentAPI(model_path, label_mapping_path, gemini_api_key)
    logger.info("IncidentVision API initialized successfully")


@app.post("/analyze/")
async def analyze_incident(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze an uploaded image for incident classification.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Complete incident analysis results
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Analyze incident
        results = api_instance.analyze_incident(image)
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/classes/")
async def get_classes():
    """Get available incident classes."""
    return {"classes": list(api_instance.label_mapping.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)