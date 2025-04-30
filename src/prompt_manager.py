"""
Prompt Management System for creating and refining effective prompts.
"""
import os
import json
import yaml
from typing import Dict, Any, List, Optional
import time

class PromptManager:
    """
    Manages the creation and optimization of prompts for different document types
    and extraction tasks. Handles prompt templates, few-shot examples, and 
    dynamic prompt generation based on document characteristics.
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the prompt manager.
        
        Args:
            storage_manager: Storage manager instance for accessing templates
        """
        self.storage_manager = storage_manager
        self.template_cache = {}
        self.load_templates()
    
    def load_templates(self):
        """Load prompt templates from storage."""
        # Default templates if none exist in storage
        self.templates = {
            "generic": {
                "understanding": {
                    "system": "You are an expert document analyst with exceptional skills in understanding document structure and content.",
                    "user": "Analyze the following text document and describe its structure, content type, and key information fields.\n\nDocument content:\n{text}\n\nProvide a detailed analysis including:\n1. Document type and purpose\n2. Main sections and their content\n3. Data fields and their formats\n4. Any patterns or recurring structures\n5. Recommendations for data extraction"
                },
                "extraction": {
                    "system": "You are a data extraction specialist with exceptional attention to detail.",
                    "user": "Extract all structured data from the following text in JSON format.\n\nDocument content:\n{text}\n\n{strategy}\n\nReturn ONLY valid JSON with no explanation or markdown formatting. Format your output as an array of objects, with each object representing a single record or data entry.\n\nFor example: [{\"field1\": \"value1\", \"field2\": \"value2\"}, {...}]\n\nEnsure all field names are consistent across all records."
                }
            },
            "financial": {
                "understanding": {
                    "system": "You are a financial document analysis expert with deep knowledge of financial statements, reports, and terminology.",
                    "user": "Analyze the following financial document and describe its structure, content type, and key financial data points.\n\nDocument content:\n{text}\n\nProvide a detailed analysis including:\n1. Financial document type (annual report, balance sheet, income statement, etc.)\n2. Main sections and their financial significance\n3. Key financial metrics and their locations\n4. Time periods represented\n5. Tables and their structure\n6. Recommendations for extracting financial data"
                },
                "extraction": {
                    "system": "You are a financial data extraction specialist with expertise in financial statements and reports.",
                    "user": "Extract all financial data from the following document in JSON format.\n\nDocument content:\n{text}\n\n{strategy}\n\nEnsure all financial figures, dates, and categories are accurately captured. Return ONLY valid JSON with no explanation or markdown formatting."
                }
            },
            "medical": {
                "understanding": {
                    "system": "You are a medical document analysis expert with knowledge of medical records, reports, and terminology.",
                    "user": "Analyze the following medical document and describe its structure, content type, and key medical information fields.\n\nDocument content:\n{text}\n\nProvide a detailed analysis including:\n1. Medical document type (patient record, lab report, clinical notes, etc.)\n2. Main sections and their medical significance\n3. Key medical data fields and their formats\n4. Patient-specific information location\n5. Test results and measurement formats\n6. Recommendations for extracting medical data"
                },
                "extraction": {
                    "system": "You are a medical data extraction specialist with expertise in healthcare documentation.",
                    "user": "Extract all medical data from the following document in JSON format.\n\nDocument content:\n{text}\n\n{strategy}\n\nEnsure all medical terms, measurements, dates, and patient information are accurately captured. Return ONLY valid JSON with no explanation or markdown formatting."
                }
            },
            "legal": {
                "understanding": {
                    "system": "You are a legal document analysis expert with knowledge of legal terminology, contract structures, and legal frameworks.",
                    "user": "Analyze the following legal document and describe its structure, content type, and key legal information fields.\n\nDocument content:\n{text}\n\nProvide a detailed analysis including:\n1. Legal document type (contract, agreement, court filing, etc.)\n2. Main sections and their legal significance\n3. Key legal terms, clauses, and definitions\n4. Parties involved and their roles\n5. Dates, deadlines, and time periods\n6. Recommendations for extracting legal data"
                },
                "extraction": {
                    "system": "You are a legal data extraction specialist with expertise in legal documentation.",
                    "user": "Extract all legal data from the following document in JSON format.\n\nDocument content:\n{text}\n\n{strategy}\n\nEnsure all legal terms, parties, dates, clauses, and conditions are accurately captured. Return ONLY valid JSON with no explanation or markdown formatting."
                }
            },
            "military_records": {
                "understanding": {
                    "system": "You are a historical military records analysis expert with knowledge of military nomenclature, ranks, and record-keeping formats.",
                    "user": "Analyze the following military record document and describe its structure, content type, and key information fields.\n\nDocument content:\n{text}\n\nProvide a detailed analysis including:\n1. Record type and historical period\n2. Main sections and their significance\n3. Key data fields (names, ranks, dates, locations, etc.)\n4. Pattern of record entries\n5. Formatting and structural elements\n6. Recommendations for extracting military record data"
                },
                "extraction": {
                    "system": "You are a historical military records extraction specialist with expertise in military documentation.",
                    "user": "Extract all individual military records from the following document in JSON format.\n\nDocument content:\n{text}\n\n{strategy}\n\nFor each record, extract the following fields (when available):\n- rank: Military rank in exact original format\n- name: Full name including first and last name\n- religion: Religious affiliation\n- marital_status: Marital status\n- district: District or region name\n- location: Specific location (village, town, etc.)\n- status: Status such as wounded, killed, missing, etc.\n- date: Date in YYYY-MM-DD format when possible\n\nReturn ONLY a valid JSON array with each record as an object. Format should be:\n[\n  {\n    \"rank\": \"...\",\n    \"name\": \"...\",\n    \"religion\": \"...\",\n    \"marital_status\": \"...\",\n    \"district\": \"...\",\n    \"location\": \"...\",\n    \"status\": \"...\",\n    \"date\": \"...\"\n  },\n  {...}\n]\n\nDo not include any markdown formatting, explanations, or non-JSON text in your response."
                }
            }
        }
        
        # Try to load saved templates if available
        try:
            saved_templates = self.storage_manager.load_templates()
            if saved_templates:
                self.templates.update(saved_templates)
        except:
            # Use default templates if loading fails
            pass
    
    def get_template(self, doc_type: str, template_type: str) -> Dict[str, str]:
        """
        Get the appropriate template for the document and template type.
        
        Args:
            doc_type: Document type (generic, financial, medical, etc.)
            template_type: Template type (understanding, extraction, etc.)
        
        Returns:
            Template dictionary with system and user prompts
        """
        # Map document types to template categories
        doc_type_map = {
            "Generic Text": "generic",
            "Financial Report": "financial",
            "Medical Record": "medical",
            "Legal Document": "legal",
            "Custom": "generic"
        }
        
        template_key = doc_type_map.get(doc_type, "generic")
        
        # Get the template
        if template_key in self.templates and template_type in self.templates[template_key]:
            return self.templates[template_key][template_type]
        
        # Fall back to generic template if specific one not found
        return self.templates["generic"][template_type]
    
    def create_understanding_prompt(self, text: str, doc_type: str) -> Dict[str, str]:
        """
        Create a prompt for initial document understanding.
        
        Args:
            text: Document text to analyze
            doc_type: Type of document
            
        Returns:
            Complete prompt with system and user content
        """
        template = self.get_template(doc_type, "understanding")
        
        # Replace placeholders
        user_prompt = template["user"].replace("{text}", text)
        
        return {
            "system": template["system"],
            "user": user_prompt
        }
    
    def create_extraction_strategy(self, understanding_result: str) -> str:
        """
        Create an extraction strategy based on document understanding.
        
        Args:
            understanding_result: Result from the document understanding step
            
        Returns:
            Extraction strategy string
        """
        strategy = "Based on the document analysis, extract data with the following approach:\n"
        
        # Try to detect key information from understanding result
        if "table" in understanding_result.lower():
            strategy += "- Extract tabular data preserving row/column relationships\n"
        
        if "date" in understanding_result.lower():
            strategy += "- Identify and standardize all dates\n"
        
        if "section" in understanding_result.lower():
            strategy += "- Preserve hierarchical section relationships\n"
        
        # Add general extraction guidance
        strategy += """
- Identify all data fields and their values
- Group related information together
- Normalize inconsistent formatting
- Provide confidence scores for uncertain extractions
- Format the extraction as a JSON object or array as appropriate
"""
        
        return strategy
    
    def create_extraction_prompt(self, text: str, doc_type: str, strategy: str) -> Dict[str, str]:
        """
        Create a prompt for data extraction.
        
        Args:
            text: Document text to extract from
            doc_type: Type of document
            strategy: Extraction strategy from document understanding
            
        Returns:
            Complete prompt with system and user content
        """
        template = self.get_template(doc_type, "extraction")
        
        # Replace placeholders
        user_prompt = template["user"].replace("{text}", text).replace("{strategy}", strategy)
        
        return {
            "system": template["system"],
            "user": user_prompt
        }
    
    def save_template(self, doc_type: str, template_type: str, template: Dict[str, str]) -> bool:
        """
        Save a new or updated template.
        
        Args:
            doc_type: Document type (generic, financial, medical, etc.)
            template_type: Template type (understanding, extraction, etc.)
            template: Template dictionary with system and user prompts
            
        Returns:
            Success status
        """
        if doc_type not in self.templates:
            self.templates[doc_type] = {}
        
        self.templates[doc_type][template_type] = template
        
        try:
            self.storage_manager.save_templates(self.templates)
            return True
        except:
            return False
    
    def create_few_shot_examples(self, doc_type: str, examples: List[Dict[str, Any]]) -> str:
        """
        Create few-shot examples section for prompts.
        
        Args:
            doc_type: Document type
            examples: List of example inputs and outputs
            
        Returns:
            Formatted few-shot examples string
        """
        few_shot_text = "Here are some examples of similar extractions:\n\n"
        
        for i, example in enumerate(examples):
            few_shot_text += f"Example {i+1}:\n"
            few_shot_text += f"Input: {example.get('input', '')}\n"
            few_shot_text += f"Output: {example.get('output', '')}\n\n"
        
        return few_shot_text 