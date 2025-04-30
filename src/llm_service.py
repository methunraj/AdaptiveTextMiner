"""
LLM Service module for interacting with Alibaba's Qwen model.
"""
import os
import time
import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
from retry import retry

class QwenService:
    """
    Service for interacting with Alibaba's Qwen model via OpenAI-compatible API.
    Provides intelligent text analysis, extraction, and pattern recognition.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Qwen service.
        
        Args:
            api_key: Optional API key. If not provided, will be set later.
        """
        self.api_key = api_key
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.default_model = "qwen-max"
        self.client = None
        self.setup_client()
        
        # Track usage stats
        self.usage_stats = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "calls_by_type": {}
        }
    
    def setup_client(self):
        """Set up the OpenAI client with current settings."""
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key.strip(),
                base_url=self.base_url
            )
    
    def set_api_key(self, api_key: str):
        """Set or update API key and reinitialize client."""
        self.api_key = api_key
        self.setup_client()
    
    @retry(tries=3, delay=2, backoff=2)
    def call_llm(self, prompt: str, params: Dict[str, Any] = None, call_type: str = "extraction") -> str:
        """
        Call the LLM with the given prompt and parameters.
        
        Args:
            prompt: The prompt to send to the LLM
            params: Optional parameters for the LLM call
            call_type: Type of call for tracking purposes
            
        Returns:
            The LLM response content
        """
        if not self.client:
            raise ValueError("API key not set. Please set API key before making calls.")
        
        # Default parameters
        default_params = {
            "model": self.default_model,
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        
        # Store document_type for internal use but don't send to API
        doc_type = None
        
        # Override defaults with provided params
        if params:
            if "document_type" in params:
                doc_type = params.pop("document_type")  # Remove and store document_type
            default_params.update(params)
        
        # Format the prompt based on call type
        if isinstance(prompt, dict):
            system_content = prompt.get("system", "You are a helpful AI assistant specialized in extracting structured data from text.")
            user_content = prompt.get("user", "")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        else:
            # If just a string, use as user content with default system prompt
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant specialized in extracting structured data from text."},
                {"role": "user", "content": prompt}
            ]
        
        start_time = time.time()
        
        try:
            # Call the API
            completion = self.client.chat.completions.create(
                messages=messages,
                **default_params
            )
            
            # Extract response
            response_text = completion.choices[0].message.content
            
            # Track usage stats
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            total_tokens = input_tokens + output_tokens
            
            # Update usage stats
            self.usage_stats["api_calls"] += 1
            self.usage_stats["input_tokens"] += input_tokens
            self.usage_stats["output_tokens"] += output_tokens
            self.usage_stats["total_tokens"] += total_tokens
            
            # Calculate cost (replace with actual Qwen pricing)
            input_cost = (input_tokens / 1000) * 0.0015  # Example price per 1K tokens
            output_cost = (output_tokens / 1000) * 0.002  # Example price per 1K tokens
            total_cost = input_cost + output_cost
            self.usage_stats["estimated_cost"] += total_cost
            
            # Track by call type
            if call_type not in self.usage_stats["calls_by_type"]:
                self.usage_stats["calls_by_type"][call_type] = {
                    "count": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            
            self.usage_stats["calls_by_type"][call_type]["count"] += 1
            self.usage_stats["calls_by_type"][call_type]["tokens"] += total_tokens
            self.usage_stats["calls_by_type"][call_type]["cost"] += total_cost
            
            return response_text
            
        except Exception as e:
            # Log the error
            print(f"Error calling LLM: {str(e)}")
            raise
    
    def extract_json_from_text(self, text: str) -> Any:
        """
        Extract valid JSON from text response.
        
        Args:
            text: Text that may contain JSON
            
        Returns:
            Parsed JSON data or None if parsing fails
        """
        if not text or not isinstance(text, str):
            print(f"Warning: Invalid text input for JSON extraction: {type(text)}")
            return None
            
        # Try to find JSON in the response
        try:
            # Look for JSON array/object pattern
            json_match = None
            
            # Try to find array
            array_start = text.find('[')
            array_end = text.rfind(']')
            if array_start >= 0 and array_end > array_start:
                json_match = text[array_start:array_end+1]
            
            # If no array, try to find object
            if not json_match:
                obj_start = text.find('{')
                obj_end = text.rfind('}')
                if obj_start >= 0 and obj_end > obj_start:
                    json_match = text[obj_start:obj_end+1]
            
            # If found potential JSON, try to parse it
            if json_match:
                parsed = json.loads(json_match)
                return parsed
            
            # If no obvious JSON delimiters found, try the whole response
            parsed = json.loads(text)
            return parsed
            
        except json.JSONDecodeError as e:
            # If JSON extraction fails, try cleaning up the text
            try:
                # Remove markdown code blocks
                clean_text = text.replace("```json", "").replace("```", "").strip()
                
                # Try more aggressive cleaning if necessary
                if not clean_text.startswith('{') and not clean_text.startswith('['):
                    # Find first occurrence of { or [
                    obj_start = clean_text.find('{')
                    array_start = clean_text.find('[')
                    
                    if obj_start >= 0 and (array_start < 0 or obj_start < array_start):
                        clean_text = clean_text[obj_start:]
                    elif array_start >= 0:
                        clean_text = clean_text[array_start:]
                
                if not clean_text.endswith('}') and not clean_text.endswith(']'):
                    # Find last occurrence of } or ]
                    obj_end = clean_text.rfind('}')
                    array_end = clean_text.rfind(']')
                    
                    if obj_end >= 0 and (array_end < 0 or obj_end > array_end):
                        clean_text = clean_text[:obj_end+1]
                    elif array_end >= 0:
                        clean_text = clean_text[:array_end+1]
                
                parsed = json.loads(clean_text)
                return parsed
            except Exception as clean_error:
                # Log the error for debugging
                print(f"JSON extraction failed: {str(e)}")
                print(f"Additional cleaning also failed: {str(clean_error)}")
                print(f"Text sample: {text[:200]}...")
                return None
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.usage_stats
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.usage_stats = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "calls_by_type": {}
        } 