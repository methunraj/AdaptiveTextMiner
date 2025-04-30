import json
import os
from typing import List, Dict, Any

def load_models_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse the models.json file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"models.json not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_providers(models_data: List[Dict[str, Any]]) -> List[str]:
    """
    Return a list of provider names.
    """
    return [provider["provider"] for provider in models_data]

def get_models_for_provider(models_data: List[Dict[str, Any]], provider: str) -> List[Dict[str, Any]]:
    """
    Return a list of models for a given provider.
    """
    for prov in models_data:
        if prov["provider"] == provider:
            return prov["models"]
    return []

def get_model_details(models_data: List[Dict[str, Any]], provider: str, model_id: str) -> Dict[str, Any]:
    """
    Return the details of a specific model given provider and model_id.
    """
    for prov in models_data:
        if prov["provider"] == provider:
            for model in prov["models"]:
                if model["model_id"] == model_id:
                    return model
    return {}
