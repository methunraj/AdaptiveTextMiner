"""
Feedback System for capturing extraction performance data.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

class FeedbackSystem:
    """
    Captures extraction performance data and user feedback to improve
    future extraction processes through prompt refinement and adaptation.
    """
    
    def __init__(self, storage_manager):
        """
        Initialize the feedback system.
        
        Args:
            storage_manager: Storage manager instance for saving feedback
        """
        self.storage_manager = storage_manager
        self.current_session = {
            "session_id": f"session_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "extractions": []
        }
    
    def record_extraction(self, doc_id: str, results_id: str, 
                         record_count: int, stats: Dict[str, Any]) -> str:
        """
        Record extraction performance metrics.
        
        Args:
            doc_id: Document ID
            results_id: Results ID
            record_count: Number of records extracted
            stats: Extraction stats
            
        Returns:
            Feedback ID
        """
        # Create feedback data
        feedback_data = {
            "document_id": doc_id,
            "results_id": results_id,
            "record_count": record_count,
            "extraction_time": datetime.now().isoformat(),
            "input_tokens": stats.get("input_tokens", 0),
            "output_tokens": stats.get("output_tokens", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "api_calls": stats.get("api_calls", 0),
            "estimated_cost": stats.get("estimated_cost", 0),
            "session_id": self.current_session["session_id"],
            "user_corrections": 0,
            "user_rating": None
        }
        
        # Add to current session
        self.current_session["extractions"].append({
            "document_id": doc_id,
            "results_id": results_id,
            "record_count": record_count,
            "extraction_time": datetime.now().isoformat()
        })
        
        # Save feedback
        feedback_id = self.storage_manager.save_feedback(feedback_data)
        
        return feedback_id
    
    def record_user_correction(self, results_id: str, 
                              correction_count: int) -> bool:
        """
        Record when user manually corrects extraction results.
        
        Args:
            results_id: Results ID
            correction_count: Number of corrections made
            
        Returns:
            Success status
        """
        # Find feedback record for this result
        feedback_files = os.listdir(self.storage_manager.feedback_dir)
        
        for filename in feedback_files:
            filepath = os.path.join(self.storage_manager.feedback_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    feedback = json.load(f)
                    
                    if feedback.get("results_id") == results_id:
                        # Update correction count
                        feedback["user_corrections"] = correction_count
                        feedback["correction_time"] = datetime.now().isoformat()
                        
                        # Save updated feedback
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(feedback, f, indent=2)
                        
                        return True
            except:
                continue
        
        return False
    
    def record_user_rating(self, results_id: str, rating: int, 
                          comments: Optional[str] = None) -> bool:
        """
        Record user satisfaction rating for extraction.
        
        Args:
            results_id: Results ID
            rating: User rating (1-5)
            comments: Optional user comments
            
        Returns:
            Success status
        """
        if rating < 1 or rating > 5:
            return False
        
        # Find feedback record for this result
        feedback_files = os.listdir(self.storage_manager.feedback_dir)
        
        for filename in feedback_files:
            filepath = os.path.join(self.storage_manager.feedback_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    feedback = json.load(f)
                    
                    if feedback.get("results_id") == results_id:
                        # Update with user rating
                        feedback["user_rating"] = rating
                        feedback["rating_time"] = datetime.now().isoformat()
                        
                        if comments:
                            feedback["user_comments"] = comments
                        
                        # Save updated feedback
                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(feedback, f, indent=2)
                        
                        return True
            except:
                continue
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated performance metrics.
        
        Returns:
            Dictionary of aggregated metrics
        """
        feedback_files = os.listdir(self.storage_manager.feedback_dir)
        
        if not feedback_files:
            return {
                "total_extractions": 0,
                "total_records": 0,
                "average_tokens_per_record": 0,
                "average_cost_per_record": 0,
                "average_user_rating": None
            }
        
        # Aggregation variables
        total_extractions = 0
        total_records = 0
        total_tokens = 0
        total_cost = 0
        ratings_sum = 0
        ratings_count = 0
        
        for filename in feedback_files:
            filepath = os.path.join(self.storage_manager.feedback_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    feedback = json.load(f)
                    
                    total_extractions += 1
                    total_records += feedback.get("record_count", 0)
                    total_tokens += feedback.get("total_tokens", 0)
                    total_cost += feedback.get("estimated_cost", 0)
                    
                    if feedback.get("user_rating") is not None:
                        ratings_sum += feedback["user_rating"]
                        ratings_count += 1
            except:
                continue
        
        # Calculate averages
        avg_tokens_per_record = total_tokens / max(total_records, 1)
        avg_cost_per_record = total_cost / max(total_records, 1)
        avg_rating = ratings_sum / ratings_count if ratings_count > 0 else None
        
        return {
            "total_extractions": total_extractions,
            "total_records": total_records,
            "average_tokens_per_record": avg_tokens_per_record,
            "average_cost_per_record": avg_cost_per_record,
            "average_user_rating": avg_rating
        }
    
    def analyze_extraction_patterns(self) -> List[Dict[str, Any]]:
        """
        Analyze extraction patterns to identify improvement opportunities.
        
        Returns:
            List of insights and improvement suggestions
        """
        feedback_files = os.listdir(self.storage_manager.feedback_dir)
        insights = []
        
        if not feedback_files:
            return insights
        
        # Document type performance
        doc_type_metrics = {}
        
        for filename in feedback_files:
            filepath = os.path.join(self.storage_manager.feedback_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    feedback = json.load(f)
                    
                    # Get document metadata to determine document type
                    doc_id = feedback.get("document_id")
                    
                    if doc_id:
                        try:
                            doc_data = self.storage_manager.get_document(doc_id)
                            doc_type = doc_data["metadata"].get("document_type", "unknown")
                            
                            if doc_type not in doc_type_metrics:
                                doc_type_metrics[doc_type] = {
                                    "extractions": 0,
                                    "records": 0,
                                    "tokens": 0,
                                    "cost": 0,
                                    "corrections": 0
                                }
                            
                            dm = doc_type_metrics[doc_type]
                            dm["extractions"] += 1
                            dm["records"] += feedback.get("record_count", 0)
                            dm["tokens"] += feedback.get("total_tokens", 0)
                            dm["cost"] += feedback.get("estimated_cost", 0)
                            dm["corrections"] += feedback.get("user_corrections", 0)
                        except:
                            pass
            except:
                continue
        
        # Generate insights
        for doc_type, metrics in doc_type_metrics.items():
            if metrics["extractions"] > 0:
                avg_records = metrics["records"] / metrics["extractions"]
                avg_tokens = metrics["tokens"] / metrics["extractions"]
                correction_rate = metrics["corrections"] / max(metrics["records"], 1)
                
                insight = {
                    "document_type": doc_type,
                    "extractions": metrics["extractions"],
                    "average_records": avg_records,
                    "correction_rate": correction_rate
                }
                
                # Add suggestion if correction rate is high
                if correction_rate > 0.1:
                    insight["suggestion"] = f"High correction rate for {doc_type} documents. Consider refining prompts or adding specialized templates."
                
                insights.append(insight)
        
        return insights
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the current session and get session summary.
        
        Returns:
            Session summary
        """
        self.current_session["end_time"] = datetime.now().isoformat()
        
        # Calculate session stats
        total_docs = len(self.current_session["extractions"])
        total_records = sum(ext.get("record_count", 0) for ext in self.current_session["extractions"])
        
        session_summary = {
            "session_id": self.current_session["session_id"],
            "start_time": self.current_session["start_time"],
            "end_time": self.current_session["end_time"],
            "documents_processed": total_docs,
            "records_extracted": total_records
        }
        
        # Save session summary
        self.storage_manager.save_feedback({
            "type": "session_summary",
            "data": session_summary
        })
        
        # Start a new session
        self.current_session = {
            "session_id": f"session_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "extractions": []
        }
        
        return session_summary
    
    def analyze_extraction_result(self, llm_service, provider, model, extracted_data, original_text, doc_type):
        """
        Analyze extraction results and suggest improvements.
        
        Args:
            llm_service: The LLM service to use for analysis
            extracted_data: The initially extracted data
            original_text: The original text chunk
            doc_type: The type of document being processed
            
        Returns:
            dict: Analysis results including quality score, issues, and suggested improvements
        """
        # Check if we have valid extracted data to analyze
        if not extracted_data or (isinstance(extracted_data, list) and len(extracted_data) == 0):
            return {
                "quality_score": 0,
                "identified_issues": ["No data was extracted in the initial pass"],
                "recommended_schema": self._get_default_schema_for_doctype(doc_type),
                "adjustment_instructions": "Look carefully for any structured data in the text. Even partial or incomplete records should be extracted.",
                "optimal_template": {"sample_field": "sample_value"}
            }
        
        # Create a summary of the extraction results for analysis
        try:
            extraction_summary = json.dumps(extracted_data, indent=2)
        except Exception as e:
            print(f"Error serializing extracted data: {e}")
            extraction_summary = str(extracted_data)[:1000]
        
        # Construct analysis prompt
        analysis_prompt = f"""
You are an expert data extraction system analyzer. Your job is to evaluate the quality of data extracted from text and suggest improvements.

DOCUMENT TYPE: {doc_type}

ORIGINAL TEXT:
```
{original_text[:2000]}{'...' if len(original_text) > 2000 else ''}
```

EXTRACTED DATA:
```
{extraction_summary}
```

Analyze the extraction results above and provide detailed feedback with the following structure:
1. Evaluate the quality of extraction on a scale of 1-10
2. Identify any issues:
   - Missing fields or data that should have been extracted
   - Formatting inconsistencies or errors
   - Incorrectly extracted values
   - Structural problems with the output
3. Suggest an improved schema for the data extraction
4. Provide specific instructions on how to adjust the extraction approach
5. Show a template of how the optimal formatting should look

Respond with a JSON object with the following structure:
{{
  "quality_score": <number 1-10>,
  "identified_issues": [<list of specific issues found>],
  "recommended_schema": {{<improved data structure>}},
  "adjustment_instructions": "<specific instructions for improving extraction>",
  "optimal_template": {{<example of ideal output format>}}
}}
"""
        
        # Call LLM for analysis
        try:
            response = llm_service.call_llm(
                provider,
                model,
                analysis_prompt,
                {
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                call_type="extraction_analysis"
            )
            
            # Parse the JSON response
            analysis_result = llm_service.extract_json_from_text(response)
            
            # Ensure the result is a dictionary with expected fields
            if not isinstance(analysis_result, dict):
                print(f"Warning: Analysis result is not a dictionary: {type(analysis_result)}")
                # Attempt to convert a list to a dictionary if possible
                if isinstance(analysis_result, list) and len(analysis_result) > 0 and isinstance(analysis_result[0], dict):
                    analysis_result = analysis_result[0]  # Use the first item
                else:
                    return self._get_default_analysis(doc_type)
                
            # Ensure all expected fields exist
            required_fields = ["quality_score", "identified_issues", "recommended_schema", 
                              "adjustment_instructions", "optimal_template"]
            for field in required_fields:
                if field not in analysis_result:
                    if field == "quality_score":
                        analysis_result[field] = 5
                    elif field == "identified_issues":
                        analysis_result[field] = ["Missing data fields"]
                    elif field in ["recommended_schema", "optimal_template"]:
                        analysis_result[field] = self._get_default_schema_for_doctype(doc_type)
                    else:
                        analysis_result[field] = "Extract all relevant information following the document structure."
            
            return analysis_result
            
        except Exception as e:
            print(f"Error analyzing extraction results: {str(e)}")
            traceback.print_exc()
            # Return a basic analysis result if an error occurs
            return self._get_default_analysis(doc_type)
            
    def _get_default_analysis(self, doc_type):
        """
        Get a default analysis structure for fallback.
        
        Args:
            doc_type: Type of document
            
        Returns:
            Default analysis dictionary
        """
        return {
            "quality_score": 5,
            "identified_issues": ["Failed to analyze extraction results"],
            "recommended_schema": self._get_default_schema_for_doctype(doc_type),
            "adjustment_instructions": "Extract all records following the standard format for this document type.",
            "optimal_template": self._get_default_schema_for_doctype(doc_type)
        }
        
    def _get_default_schema_for_doctype(self, doc_type):
        """
        Get a default schema based on document type.
        
        Args:
            doc_type: Type of document
            
        Returns:
            Default schema dictionary
        """
        if doc_type == "Military Records":
            return {
                "rank": "string",
                "name": "string",
                "religion": "string",
                "marital_status": "string",
                "district": "string",
                "location": "string",
                "status": "string",
                "date": "string"
            }
        elif doc_type == "Financial Report":
            return {
                "item": "string",
                "amount": "number",
                "category": "string", 
                "date": "string",
                "notes": "string"
            }
        elif doc_type == "Medical Record":
            return {
                "patient_id": "string",
                "procedure": "string",
                "date": "string",
                "practitioner": "string",
                "notes": "string"
            }
        else:
            # Generic schema
            return {
                "field1": "string",
                "field2": "string",
                "field3": "string",
                "date": "string"
            }
    
    def adaptive_extraction_prompt(self, llm_service, provider, model, original_prompt, initial_results, text_chunk, doc_type):
        """
        Generate an improved extraction prompt based on analysis of initial results.
        
        Args:
            llm_service: The LLM service to use for extraction
            original_prompt: The prompt used for initial extraction (optional)
            initial_results: The results from the initial extraction
            text_chunk: The text chunk to extract data from
            doc_type: The type of document being processed
            
        Returns:
            str: An improved prompt for extraction
        """
        # First, analyze the initial extraction results
        analysis = self.analyze_extraction_result(
            llm_service,
            provider,
            model,
            initial_results,
            text_chunk,
            doc_type
        )
        
        # Ensure analysis is a dictionary
        if not isinstance(analysis, dict):
            print(f"Warning: Analysis result is not a dictionary: {type(analysis)}")
            analysis = {
                "quality_score": 5,
                "identified_issues": ["Analysis returned in incorrect format"],
                "recommended_schema": {},
                "adjustment_instructions": "Extract all relevant information following the document structure.",
                "optimal_template": {}
            }
        
        # Create an adaptive prompt based on the analysis
        adaptive_prompt = f"""
You are an expert data extraction system specializing in {doc_type} documents.

Your task is to extract structured data from the following text, optimizing for:
1. Completeness - extract ALL relevant information
2. Accuracy - ensure the extracted data matches the original text
3. Consistency - maintain consistent formatting across fields
4. Structure - follow the recommended schema exactly

DOCUMENT TYPE: {doc_type}

ANALYSIS OF PREVIOUS EXTRACTION:
Quality Score: {analysis.get('quality_score', 'N/A')}
Identified Issues: {', '.join(analysis.get('identified_issues', ['No issues identified']))}

EXTRACTION ADJUSTMENT INSTRUCTIONS:
{analysis.get('adjustment_instructions', 'Extract all relevant information following the document structure.')}

RECOMMENDED SCHEMA TO FOLLOW:
{json.dumps(analysis.get('recommended_schema', {}), indent=2)}

OPTIMAL FORMATTING:
{json.dumps(analysis.get('optimal_template', {}), indent=2)}

TEXT TO EXTRACT FROM:
```
{text_chunk}
```

Respond ONLY with the extracted data in JSON format following the recommended schema exactly.
Do not include any explanations, introductions, or notes in your response.
Include ALL fields in the schema, using null for missing values.

IMPORTANT: If you cannot find any structured data to extract, return a JSON array with a single object that has a "status" field with value "no_data_found" and a "message" field explaining why no data could be extracted.
Example: [{{"status": "no_data_found", "message": "No structured records found in the provided text."}}]
"""
        
        return adaptive_prompt 