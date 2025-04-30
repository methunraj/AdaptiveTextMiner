import os
import time
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from retry import retry

# DeepSeek and others can be added similarly

class MultiLLMService:
    """
    Service for interacting with multiple LLM providers/models using models.json metadata.
    """
    def __init__(self, models_data):
        self.models_data = models_data
        self.clients = {}  # Cache for provider/model-specific clients
        self.usage_stats = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "calls_by_type": {}
        }

    def get_api_key(self, provider: str):
        import re
        # Remove non-alphanumeric characters except underscores
        clean_provider = re.sub(r'[^A-Z0-9_]', '', provider.upper().replace(' ', '_'))
        env_var = f"{clean_provider}_API_KEY"
        api_key = os.getenv(env_var, "")
        # Fallback for Qwen (Alibaba) to QWEN_API_KEY
        if not api_key and 'QWEN' in clean_provider:
            api_key = os.getenv('QWEN_API_KEY', "")
        return api_key

    def get_client(self, provider: str, model: Dict[str, Any], api_key: str):
        if provider == "OpenAI" or model.get("api_type") == "chat_completions":
            base_url = model.get("endpoint") or "https://api.openai.com/v1"
            return OpenAI(api_key=api_key.strip(), base_url=base_url)
        elif provider == "Qwen (Alibaba)" or model.get("api_type") == "openai_compatible":
            base_url = model.get("endpoint") or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            return OpenAI(api_key=api_key.strip(), base_url=base_url)
        elif provider == "DeepSeek":
            # Placeholder: DeepSeek is OpenAI-compatible, but may need custom endpoint
            base_url = model.get("endpoint") or "https://api.deepseek.com/v1"
            return OpenAI(api_key=api_key.strip(), base_url=base_url)
        raise NotImplementedError(f"Provider {provider} not yet supported.")

    @retry(tries=3, delay=2, backoff=2)
    def call_llm(self, provider: str, model: Dict[str, Any], prompt: Any, params: Optional[Dict[str, Any]] = None, call_type: str = "extraction", api_key: Optional[str] = None) -> str:
        """
        Call the selected LLM with the given prompt and parameters.
        """
        import time
        if not api_key:
            api_key = self.get_api_key(provider)
        if not api_key:
            raise ValueError(f"API key for {provider} not set.")
        client = self.get_client(provider, model, api_key)
        model_id = model["model_id"]
        default_params = {
            "model": model_id,
            "temperature": params.get("temperature", 0.1) if params else 0.1,
            "max_tokens": params.get("max_tokens", model.get("max_output_tokens", 4000)) if params else model.get("max_output_tokens", 4000)
        }
        # Make a copy of params to avoid modifying the original
        if params:
            params = dict(params)
            # Remove non-API parameters
            if "document_type" in params:
                params.pop("document_type")
            if "schema_id" in params:
                params.pop("schema_id")
        if params:
            default_params.update(params)
        start_time = time.time()
        response_text = None
        input_tokens = 0
        output_tokens = 0
        # OpenAI-compatible (OpenAI, Qwen, DeepSeek)
        if provider in ["OpenAI", "Qwen (Alibaba)", "DeepSeek"] or model.get("api_type") in ["chat_completions", "openai_compatible"]:
            if isinstance(prompt, dict):
                system_content = prompt.get("system", "You are a helpful AI assistant specialized in extracting structured data from text.")
                user_content = prompt.get("user", "")
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant specialized in extracting structured data from text."},
                    {"role": "user", "content": prompt}
                ]
            completion = client.chat.completions.create(
                messages=messages,
                **default_params
            )
            response_text = completion.choices[0].message.content
            input_tokens = getattr(completion.usage, 'prompt_tokens', 0)
            output_tokens = getattr(completion.usage, 'completion_tokens', 0)
        else:
            raise NotImplementedError(f"Provider {provider} not yet supported.")
        # Usage/cost tracking
        total_tokens = input_tokens + output_tokens
        self.usage_stats["api_calls"] += 1
        self.usage_stats["input_tokens"] += input_tokens
        self.usage_stats["output_tokens"] += output_tokens
        self.usage_stats["total_tokens"] += total_tokens
        # Pricing from models.json
        pricing = model.get("pricing", {})
        input_cost = (input_tokens / 1_000_000) * float(pricing.get("input_per_1M", 0))
        output_cost = (output_tokens / 1_000_000) * float(pricing.get("output_per_1M", 0))
        total_cost = input_cost + output_cost
        self.usage_stats["estimated_cost"] += total_cost
        if call_type not in self.usage_stats["calls_by_type"]:
            self.usage_stats["calls_by_type"][call_type] = {"count": 0, "tokens": 0, "cost": 0.0}
        self.usage_stats["calls_by_type"][call_type]["count"] += 1
        self.usage_stats["calls_by_type"][call_type]["tokens"] += total_tokens
        self.usage_stats["calls_by_type"][call_type]["cost"] += total_cost
        return response_text

    def extract_json_from_text(self, text: str) -> Any:
        """
        Extract valid JSON from text response.
        """
        if not text or not isinstance(text, str):
            return None
        try:
            array_start = text.find('[')
            array_end = text.rfind(']')
            if array_start >= 0 and array_end > array_start:
                json_match = text[array_start:array_end+1]
            else:
                obj_start = text.find('{')
                obj_end = text.rfind('}')
                if obj_start >= 0 and obj_end > obj_start:
                    json_match = text[obj_start:obj_end+1]
                else:
                    json_match = text
            return json.loads(json_match)
        except Exception:
            try:
                clean_text = text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_text)
            except Exception:
                return None

    def get_usage_stats(self) -> dict:
        return self.usage_stats

    def reset_usage_stats(self):
        self.usage_stats = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "calls_by_type": {}
        }
