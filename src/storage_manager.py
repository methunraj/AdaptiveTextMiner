"""
Storage Manager for handling document and result storage.
"""
import os
import json
import yaml
import pandas as pd
import io
from datetime import datetime
from typing import Dict, Any, List, Optional, BinaryIO, Union
import uuid

class StorageManager:
    """
    Manages persistent storage of documents, extraction results, and system data.
    Handles saving, retrieving, and organizing data with appropriate metadata.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the storage manager.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir
        self.documents_dir = os.path.join(base_dir, "documents")
        self.results_dir = os.path.join(base_dir, "results")
        self.prompts_dir = os.path.join(base_dir, "prompts")
        self.feedback_dir = os.path.join(base_dir, "feedback")
        self.schemas_dir = os.path.join(base_dir, "schemas")
        
        # Ensure directories exist
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.base_dir, self.documents_dir, self.results_dir,
                         self.prompts_dir, self.feedback_dir, self.schemas_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_document(self, filename: str, content: str, doc_type: str) -> str:
        """
        Save a document with metadata.
        
        Args:
            filename: Original filename
            content: Document content
            doc_type: Document type
            
        Returns:
            Document ID
        """
        # Generate a unique ID
        doc_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "id": doc_id,
            "original_filename": filename,
            "document_type": doc_type,
            "upload_time": datetime.now().isoformat(),
            "file_size_bytes": len(content),
            "character_count": len(content),
            "line_count": content.count('\n') + 1
        }
        
        # Create document directory
        doc_dir = os.path.join(self.documents_dir, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Save content and metadata
        with open(os.path.join(doc_dir, "content.txt"), "w", encoding="utf-8") as f:
            f.write(content)
        
        with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a document and its metadata.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary with document content and metadata
        """
        doc_dir = os.path.join(self.documents_dir, doc_id)
        
        if not os.path.exists(doc_dir):
            raise FileNotFoundError(f"Document with ID {doc_id} not found")
        
        # Load content
        with open(os.path.join(doc_dir, "content.txt"), "r", encoding="utf-8") as f:
            content = f.read()
        
        # Load metadata
        with open(os.path.join(doc_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    def save_results(self, doc_id: str, results: List[Dict[str, Any]]) -> str:
        """
        Save extraction results with metadata.
        
        Args:
            doc_id: Associated document ID
            results: Extraction results
            
        Returns:
            Results ID
        """
        # Generate a unique ID
        results_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "id": results_id,
            "document_id": doc_id,
            "extraction_time": datetime.now().isoformat(),
            "record_count": len(results)
        }
        
        # Create results directory
        results_dir = os.path.join(self.results_dir, results_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results and metadata
        with open(os.path.join(results_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(results_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Save as CSV for easy import
        try:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_dir, "results.csv"), index=False, encoding="utf-8")
        except:
            # Skip CSV if conversion fails
            pass
        
        return results_id
    
    def get_results(self, results_id: str) -> Dict[str, Any]:
        """
        Retrieve extraction results and metadata.
        
        Args:
            results_id: Results ID
            
        Returns:
            Dictionary with results and metadata
        """
        results_dir = os.path.join(self.results_dir, results_id)
        
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Results with ID {results_id} not found")
        
        # Load results
        with open(os.path.join(results_dir, "results.json"), "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Load metadata
        with open(os.path.join(results_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return {
            "results": results,
            "metadata": metadata
        }
    
    def save_templates(self, templates: Dict[str, Any]) -> bool:
        """
        Save prompt templates.
        
        Args:
            templates: Dictionary of templates
            
        Returns:
            Success status
        """
        try:
            templates_file = os.path.join(self.prompts_dir, "templates.json")
            with open(templates_file, "w", encoding="utf-8") as f:
                json.dump(templates, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
            return False
            
    def save_schema(self, schema_name: str, schema_data: Dict[str, Any], doc_type: str = "Generic") -> str:
        """
        Save a data extraction schema.
        
        Args:
            schema_name: Name of the schema
            schema_data: Schema definition including fields and types
            doc_type: Document type this schema is designed for
            
        Returns:
            Schema ID
        """
        # Generate a unique ID
        schema_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "id": schema_id,
            "name": schema_name,
            "document_type": doc_type,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "field_count": len(schema_data.get("fields", []))
        }
        
        # Create schema directory
        schema_dir = os.path.join(self.schemas_dir, schema_id)
        os.makedirs(schema_dir, exist_ok=True)
        
        # Save schema and metadata
        with open(os.path.join(schema_dir, "schema.json"), "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(schema_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        return schema_id
    
    def get_schema(self, schema_id: str) -> Dict[str, Any]:
        """
        Retrieve a schema and its metadata.
        
        Args:
            schema_id: Schema ID
            
        Returns:
            Dictionary with schema definition and metadata
        """
        schema_dir = os.path.join(self.schemas_dir, schema_id)
        
        if not os.path.exists(schema_dir):
            raise FileNotFoundError(f"Schema with ID {schema_id} not found")
        
        # Load schema
        with open(os.path.join(schema_dir, "schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)
        
        # Load metadata
        with open(os.path.join(schema_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        return {
            "schema": schema,
            "metadata": metadata
        }
    
    def list_schemas(self, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available schemas, optionally filtered by document type.
        
        Args:
            doc_type: Optional document type filter
            
        Returns:
            List of schema metadata
        """
        schemas = []
        
        if not os.path.exists(self.schemas_dir):
            return schemas
        
        for schema_id in os.listdir(self.schemas_dir):
            schema_dir = os.path.join(self.schemas_dir, schema_id)
            
            if os.path.isdir(schema_dir):
                try:
                    # Load metadata
                    with open(os.path.join(schema_dir, "metadata.json"), "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                    
                    # Filter by document type if specified
                    if doc_type is None or metadata.get("document_type") == doc_type:
                        schemas.append(metadata)
                except Exception as e:
                    print(f"Error loading schema metadata for {schema_id}: {e}")
        
        # Sort by creation date (newest first)
        schemas.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return schemas
    
    def delete_schema(self, schema_id: str) -> bool:
        """
        Delete a schema.
        
        Args:
            schema_id: Schema ID to delete
            
        Returns:
            Success status
        """
        schema_dir = os.path.join(self.schemas_dir, schema_id)
        
        if not os.path.exists(schema_dir):
            return False
        
        try:
            import shutil
            shutil.rmtree(schema_dir)
            return True
        except Exception as e:
            print(f"Error deleting schema {schema_id}: {e}")
            return False
    
    def load_templates(self) -> Optional[Dict[str, Any]]:
        """
        Load prompt templates.
        
        Returns:
            Dictionary of templates or None if not found
        """
        templates_file = os.path.join(self.prompts_dir, "templates.json")
        
        if not os.path.exists(templates_file):
            return None
        
        try:
            with open(templates_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading templates: {str(e)}")
            return None
            
    def get_as_excel(self, df: pd.DataFrame) -> bytes:
        """
        Convert a DataFrame to Excel bytes for download.
        
        Args:
            df: Pandas DataFrame to convert
            
        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Extraction Results')
            # Auto-adjust column widths
            worksheet = writer.sheets['Extraction Results']
            for i, col in enumerate(df.columns):
                max_width = max(
                    df[col].astype(str).map(len).max(),
                    len(col)
                ) + 2  # padding
                # Convert to Excel column width which is in characters
                worksheet.column_dimensions[chr(65 + i)].width = min(max_width, 50)  # limit to 50 char width
        
        output.seek(0)
        return output.getvalue()
        
    def save_batch_results(self, batch_results: List[Dict[str, Any]], output_dir: str = None) -> Dict[str, str]:
        """
        Save batch extraction results to a specified directory.
        
        Args:
            batch_results: List of dictionaries with filename and results
            output_dir: Directory to save results (if None, use default results dir)
            
        Returns:
            Dictionary mapping filenames to result IDs
        """
        if output_dir is None:
            output_dir = self.results_dir
        else:
            os.makedirs(output_dir, exist_ok=True)
            
        result_ids = {}
        
        # Process each file's results
        for item in batch_results:
            filename = item.get('filename', 'unknown')
            doc_id = item.get('doc_id', str(uuid.uuid4()))
            results = item.get('results', [])
            
            if not results:
                continue
                
            # Save results
            results_id = self.save_results(doc_id, results)
            result_ids[filename] = results_id
            
            # Create a dataframe
            try:
                df = pd.DataFrame(results)
                
                # Save as Excel in output directory
                clean_filename = os.path.splitext(os.path.basename(filename))[0]
                excel_path = os.path.join(output_dir, f"{clean_filename}_results.xlsx")
                
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Extraction Results')
                    # Auto-adjust column widths
                    worksheet = writer.sheets['Extraction Results']
                    for i, col in enumerate(df.columns):
                        max_width = max(
                            df[col].astype(str).map(len).max(),
                            len(col)
                        ) + 2  # padding
                        # Convert to Excel column width which is in characters
                        worksheet.column_dimensions[chr(65 + i)].width = min(max_width, 50)
            except Exception as e:
                print(f"Error saving Excel for {filename}: {str(e)}")
                
        return result_ids
        
    def extract_and_save_from_zip(self, zip_file: BinaryIO, processor_func, output_dir: str = None, **kwargs) -> Dict[str, Any]:
        """
        Extract text files from a zip, process each, and save results.
        
        Args:
            zip_file: Uploaded zip file object
            processor_func: Function to process each text file
            output_dir: Directory to save results (default: results_dir)
            **kwargs: Additional arguments for processor_func
            
        Returns:
            Summary statistics of processing
        """
        import zipfile
        import tempfile
        
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(output_dir, exist_ok=True)
            
        stats = {
            'total_files': 0,
            'processed_files': 0, 
            'skipped_files': 0,
            'error_files': 0,
            'results': []
        }
        
        # Create a temporary directory to extract files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the zip file to temp
            zip_path = os.path.join(temp_dir, 'archive.zip')
            with open(zip_path, 'wb') as f:
                f.write(zip_file.read())
                
            # Extract the zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
                # Process each text file
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.lower().endswith('.txt'):
                            stats['total_files'] += 1
                            file_path = os.path.join(root, file)
                            
                            try:
                                # Read the text file
                                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                    text_content = f.read()
                                    
                                # Process the file
                                try:
                                    # Save document and get doc_id
                                    doc_id = self.save_document(file, text_content, kwargs.get('doc_type', 'Generic Text'))
                                    
                                    # Process with the provided function
                                    results = processor_func(
                                        text_content=text_content,
                                        filename=file,
                                        doc_id=doc_id,
                                        **kwargs
                                    )
                                    
                                    # Add to results
                                    stats['results'].append({
                                        'filename': file,
                                        'doc_id': doc_id,
                                        'results': results,
                                        'status': 'success'
                                    })
                                    stats['processed_files'] += 1
                                    
                                except Exception as e:
                                    stats['error_files'] += 1
                                    stats['results'].append({
                                        'filename': file,
                                        'status': 'error',
                                        'error': str(e)
                                    })
                            except Exception as e:
                                stats['error_files'] += 1
                                stats['results'].append({
                                    'filename': file,
                                    'status': 'error',
                                    'error': f"File reading error: {str(e)}"
                                })
                        else:
                            stats['skipped_files'] += 1
        
        # Save batch results to the output directory
        result_ids = self.save_batch_results(
            [r for r in stats['results'] if r.get('status') == 'success'],
            output_dir
        )
        
        # Add result_ids to stats
        stats['result_ids'] = result_ids
        stats['output_dir'] = output_dir
        
        return stats
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Save extraction feedback data.
        
        Args:
            feedback_data: Feedback data dictionary
            
        Returns:
            Feedback ID
        """
        # Generate a unique ID
        feedback_id = str(uuid.uuid4())
        feedback_data["id"] = feedback_id
        feedback_data["timestamp"] = datetime.now().isoformat()
        
        # Save feedback
        feedback_file = os.path.join(self.feedback_dir, f"{feedback_id}.json")
        
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2)
        
        return feedback_id
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all documents with metadata.
        
        Returns:
            List of document metadata
        """
        documents = []
        
        for doc_id in os.listdir(self.documents_dir):
            doc_dir = os.path.join(self.documents_dir, doc_id)
            
            if os.path.isdir(doc_dir):
                metadata_file = os.path.join(doc_dir, "metadata.json")
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            documents.append(metadata)
                    except:
                        pass
        
        return documents
    
    def get_results_list(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of all results with metadata, optionally filtered by document ID.
        
        Args:
            doc_id: Optional document ID to filter by
            
        Returns:
            List of results metadata
        """
        results_list = []
        
        for results_id in os.listdir(self.results_dir):
            results_dir = os.path.join(self.results_dir, results_id)
            
            if os.path.isdir(results_dir):
                metadata_file = os.path.join(results_dir, "metadata.json")
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r", encoding="utf-8") as f:
                            metadata = json.load(f)
                            
                            if doc_id is None or metadata.get("document_id") == doc_id:
                                results_list.append(metadata)
                    except:
                        pass
        
        return results_list
    
    def get_as_excel(self, df: pd.DataFrame) -> bytes:
        """
        Convert a DataFrame to Excel bytes for download.
        
        Args:
            df: DataFrame to convert
            
        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Extraction Results', index=False)
        
        output.seek(0)
        return output.getvalue()
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its associated results.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Success status
        """
        doc_dir = os.path.join(self.documents_dir, doc_id)
        
        if not os.path.exists(doc_dir):
            return False
        
        try:
            # Delete document files
            for filename in os.listdir(doc_dir):
                os.remove(os.path.join(doc_dir, filename))
            
            # Remove document directory
            os.rmdir(doc_dir)
            
            # Delete associated results
            for results_meta in self.get_results_list(doc_id):
                results_id = results_meta.get("id")
                if results_id:
                    self.delete_results(results_id)
            
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False
    
    def delete_results(self, results_id: str) -> bool:
        """
        Delete extraction results.
        
        Args:
            results_id: Results ID
            
        Returns:
            Success status
        """
        results_dir = os.path.join(self.results_dir, results_id)
        
        if not os.path.exists(results_dir):
            return False
        
        try:
            # Delete results files
            for filename in os.listdir(results_dir):
                os.remove(os.path.join(results_dir, filename))
            
            # Remove results directory
            os.rmdir(results_dir)
            
            return True
        except Exception as e:
            print(f"Error deleting results: {str(e)}")
            return False 