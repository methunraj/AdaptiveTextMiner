import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
import time
import yaml
from datetime import datetime
import plotly.express as px
from dotenv import load_dotenv
import io

from src.multi_llm_service import MultiLLMService
from src.prompt_manager import PromptManager
from src.context_manager import ContextManager
from src.feedback_system import FeedbackSystem
from src.storage_manager import StorageManager
from src.model_registry import load_models_json, get_providers, get_models_for_provider, get_model_details

# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(page_title="Text Extraction System by Qwen/Deepseek/OpenAI", page_icon="📄", layout="wide")

# Initialize services
@st.cache_resource
def initialize_services(models_data):
    storage_manager = StorageManager("data")
    prompt_manager = PromptManager(storage_manager)
    llm_service = MultiLLMService(models_data)
    context_manager = ContextManager()
    feedback_system = FeedbackSystem(storage_manager)
    return {
        "storage_manager": storage_manager,
        "prompt_manager": prompt_manager,
        "llm_service": llm_service,
        "context_manager": context_manager,
        "feedback_system": feedback_system
    }

# Add schema management methods to StorageManager if they don't exist
def ensure_schema_methods(storage_manager):
    import os
    import json
    import uuid
    from datetime import datetime
    from typing import Dict, Any, List, Optional
    
    # Create schemas directory if it doesn't exist
    schemas_dir = os.path.join(storage_manager.base_dir, "schemas")
    os.makedirs(schemas_dir, exist_ok=True)
    
    # Add schemas_dir attribute if it doesn't exist
    if not hasattr(storage_manager, "schemas_dir"):
        storage_manager.schemas_dir = schemas_dir
    
    # Add save_schema method if it doesn't exist
    if not hasattr(storage_manager, "save_schema"):
        def save_schema(self, schema_name: str, schema_data: Dict[str, Any], doc_type: str = "Generic") -> str:
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
        
        storage_manager.save_schema = save_schema.__get__(storage_manager)
    
    # Add get_schema method if it doesn't exist
    if not hasattr(storage_manager, "get_schema"):
        def get_schema(self, schema_id: str) -> Dict[str, Any]:
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
        
        storage_manager.get_schema = get_schema.__get__(storage_manager)
    
    # Add list_schemas method if it doesn't exist
    if not hasattr(storage_manager, "list_schemas"):
        def list_schemas(self, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
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
        
        storage_manager.list_schemas = list_schemas.__get__(storage_manager)
    
    # Add delete_schema method if it doesn't exist
    if not hasattr(storage_manager, "delete_schema"):
        def delete_schema(self, schema_id: str) -> bool:
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
        
        storage_manager.delete_schema = delete_schema.__get__(storage_manager)

# Load and parse models.json
MODELS_JSON_PATH = os.path.join(os.path.dirname(__file__), "models.json") if os.path.exists(os.path.join(os.path.dirname(__file__), "models.json")) else os.path.join(os.getcwd(), "models.json")
models_data = load_models_json(MODELS_JSON_PATH)
services = initialize_services(models_data)

# Ensure schema management methods are available
ensure_schema_methods(services["storage_manager"])

# Sidebar
with st.sidebar:
    st.title("📄 Text Extraction System")
    st.subheader("Multi-Provider & Model Support")

    # Provider and Model Selection
    providers = get_providers(models_data)
    provider = st.selectbox("Provider", providers)
    available_models = get_models_for_provider(models_data, provider)
    model_display_names = [m["display_name"] for m in available_models]
    model_idx = 0
    if len(model_display_names) > 0:
        model_idx = 0
    selected_model_display = st.selectbox("Model", model_display_names, index=model_idx)
    selected_model = next((m for m in available_models if m["display_name"] == selected_model_display), available_models[0])

    # Show model details
    st.markdown(f"**Description:** {selected_model.get('description', '-')}")
    st.markdown(f"**Context Window:** {selected_model.get('context_window', '-')}")
    st.markdown(f"**Max Output Tokens:** {selected_model.get('max_output_tokens', '-')}")
    st.markdown(f"**Knowledge Cutoff:** {selected_model.get('knowledge_cutoff', '-')}")
    pricing = selected_model.get('pricing', {})
    st.markdown(f"**Pricing:** Input: ${pricing.get('input_per_1M', '-')}/1M, Output: ${pricing.get('output_per_1M', '-')}/1M")
    st.markdown(f"**Features:** {', '.join(selected_model.get('features', []))}")
    st.markdown(f"**API Type:** {selected_model.get('api_type', '-')}")

    # API Key Input (per provider)
    import re
    clean_provider = re.sub(r'[^A-Z0-9_]', '', provider.upper().replace(' ', '_'))
    api_key_env_var = f"{clean_provider}_API_KEY"
    api_key = os.getenv(api_key_env_var, "")
    # Fallback for Qwen (Alibaba) to QWEN_API_KEY
    if not api_key and 'QWEN' in clean_provider:
        api_key = os.getenv('QWEN_API_KEY', "")
    if not api_key:
        api_key = st.text_input(f"{provider} API Key", type="password")
        # You may want to store this in session_state for later use
        if api_key:
            st.session_state[api_key_env_var] = api_key
            st.success("API Key set!")
    else:
        st.success("API Key loaded from environment!")

    # API Connection Test Button
    if st.button("Test API Connection", key="test_api_conn"):
        try:
            test_prompt = "Say hello. Respond with a short greeting."
            test_response = services["llm_service"].call_llm(
                provider=provider,
                model=selected_model,
                prompt=test_prompt,
                params={"max_tokens": 20, "temperature": 0.0},
                api_key=api_key
            )
            st.success(f"API Connection Successful! Response: {test_response}")
        except Exception as e:
            st.error(f"API Connection Failed: {e}")

    # Document Type Selection
    default_doc_type = os.getenv("DEFAULT_DOC_TYPE", "Generic Text")
    doc_type = st.selectbox(
        "Document Type",
        ["Generic Text", "Financial Report", "Medical Record", "Legal Document", "Military Records", "Custom"],
        index=["Generic Text", "Financial Report", "Medical Record", "Legal Document", "Military Records", "Custom"].index(default_doc_type) if default_doc_type in ["Generic Text", "Financial Report", "Medical Record", "Legal Document", "Military Records", "Custom"] else 0
    )
    
    # Schema Selection
    try:
        # Try to get schemas if the method is available
        schemas = services["storage_manager"].list_schemas(doc_type)
        schema_options = ["None (Auto-detect)"] + [f"{s['name']} ({s['field_count']} fields)" for s in schemas]
    except (AttributeError, Exception) as e:
        # Fallback if method is not available
        st.warning("Schema management is not available. The feature might need to be reloaded.")
        schemas = []
        schema_options = ["None (Auto-detect)"]
    
    selected_schema_option = st.selectbox(
        "Use Extraction Schema",
        options=schema_options,
        index=0,
        help="Select a predefined schema to structure the extraction results"
    )
    
    # Get the selected schema ID if one was selected
    selected_schema_id = None
    selected_schema = None
    if selected_schema_option != "None (Auto-detect)" and schemas:
        selected_schema_name = selected_schema_option.split(" (")[0]
        for schema in schemas:
            if schema["name"] == selected_schema_name:
                selected_schema_id = schema["id"]
                # Load the full schema details
                try:
                    selected_schema = services["storage_manager"].get_schema(selected_schema_id)["schema"]
                    st.success(f"Using schema: {selected_schema_name}")
                except Exception as e:
                    st.warning(f"Could not load schema details: {str(e)}")
                break

    # Advanced Options Expander
    with st.expander("Advanced Options", expanded=False):
        # Get model metadata for dynamic UI
        model_ctx = selected_model.get('context_window', 4096)
        model_max_tokens = selected_model.get('max_output_tokens', 2048)
        model_temp_min = selected_model.get('temperature_min', 0.0)
        model_temp_max = selected_model.get('temperature_max', 1.0)
        model_temp_default = selected_model.get('temperature_default', 0.1)
        # Sliders dynamically set by model.json
        temperature = st.slider(
            "Temperature",
            float(model_temp_min), float(model_temp_max), float(model_temp_default), 0.01,
            help="Controls randomness: lower is more deterministic, higher is more creative."
        )
        max_tokens = st.slider(
            "Max Output Tokens",
            min(256, int(model_max_tokens)),
            int(model_max_tokens),
            int(model_max_tokens),
            32,
            help="Maximum tokens in the model's output. Limited by the model's max tokens."
        )
        context_window = st.number_input(
            "Context Window (tokens)",
            min_value=512,
            max_value=int(model_ctx),
            value=int(model_ctx),
            step=128,
            help="Maximum number of tokens the model can consider in context."
        )
        chunk_size = st.slider(
            "Chunk Size (chars)",
            1000, 50000,
            15000,
            1000,
            help="Document chunk size for extraction."
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (chars)",
            0, 5000,
            1000,
            100,
            help="Overlap between chunks for context continuity."
        )
        # Show all model metadata as reference
        st.markdown("---")
        st.caption("**Model Metadata (from models.json):**")
        st.json(selected_model)

    # Custom Prompt Input
    custom_prompt = st.text_area(
        "Custom Extraction Prompt (Optional)",
        value="",
        help="Provide specific instructions for how you want the extracted data to look or be formatted. If provided, the AI will refine this prompt after analyzing your document before extraction."
    )
    optimized_prompt = None

    # Processing Button
    process_button = st.button("Process Document", type="primary", disabled=not api_key)

# Main content
st.title("Advanced LLM Text Extraction System")

# Create tabs for different functionality
tab1, tab2 = st.tabs(["📄 Document Processing", "🔍 Schema Management"])

# Tab 1: Document Processing
with tab1:
    # Usage & Cost Dashboard
    with st.expander("💸 Usage & Cost Dashboard", expanded=True):
        usage_stats = services["llm_service"].get_usage_stats()
        st.markdown(f"**API Calls:** {usage_stats['api_calls']}")
        st.markdown(f"**Input Tokens:** {usage_stats['input_tokens']}")
        st.markdown(f"**Output Tokens:** {usage_stats['output_tokens']}")
        st.markdown(f"**Total Tokens:** {usage_stats['total_tokens']}")
        st.markdown(f"**Estimated Cost:** ${usage_stats['estimated_cost']:.6f}")
        st.caption("Pricing is dynamically loaded from models.json. Adjust prices there and the dashboard will auto-update.")

# Tab 2: Schema Management
with tab2:
    st.header("Schema Management")
    st.markdown("""
    Create and manage data extraction schemas to standardize your extraction results. 
    Schemas define the structure of the data you want to extract from documents.
    """)
    
    # Schema creation form
    with st.expander("Create New Schema", expanded=True):
        schema_form = st.form("schema_form")
        with schema_form:
            schema_name = st.text_input("Schema Name", placeholder="e.g., Military Personnel Records")
            schema_doc_type = st.selectbox(
                "Document Type",
                ["Generic Text", "Financial Report", "Medical Record", "Legal Document", "Military Records", "Custom"],
                index=0
            )
            st.markdown("#### Define Schema Fields")
            st.markdown("Enter field definitions in JSON format. Each field should include a name, type, and optional description.")
            
            schema_template = {
                "fields": [
                    {"name": "field1", "type": "string", "description": "Description of field1"},
                    {"name": "field2", "type": "number", "description": "Description of field2"}
                ]
            }
            
            schema_json = st.text_area(
                "Schema JSON", 
                value=json.dumps(schema_template, indent=2),
                height=300
            )
            
            submit_schema = st.form_submit_button("Save Schema")
        
        if submit_schema:
            try:
                # Parse and validate schema JSON
                schema_data = json.loads(schema_json)
                
                # Basic validation
                if not isinstance(schema_data, dict) or "fields" not in schema_data or not isinstance(schema_data["fields"], list):
                    st.error("Invalid schema format. Schema must contain a 'fields' array.")
                elif not schema_name:
                    st.error("Schema name is required.")
                else:
                    # Save schema
                    schema_id = services["storage_manager"].save_schema(schema_name, schema_data, schema_doc_type)
                    st.success(f"Schema '{schema_name}' saved successfully!")
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your schema definition.")
            except Exception as e:
                st.error(f"Error saving schema: {str(e)}")
    
    # List existing schemas
    st.subheader("Available Schemas")
    try:
        schemas = services["storage_manager"].list_schemas()
        
        if not schemas:
            st.info("No schemas found. Create your first schema using the form above.")
        else:
            # Create a table of schemas
            schema_data = []
            for schema in schemas:
                schema_data.append({
                    "Name": schema.get("name", "Unnamed"),
                    "Document Type": schema.get("document_type", "Unknown"),
                    "Fields": schema.get("field_count", 0),
                    "Created": schema.get("created_at", "").split("T")[0],
                    "ID": schema.get("id", "")
                })
            
            schema_df = pd.DataFrame(schema_data)
            st.dataframe(schema_df.drop(columns=["ID"]), use_container_width=True)
            
            # Schema details and actions
            selected_schema_id = st.selectbox("Select a schema to view or manage", 
                                            options=schema_df["ID"].tolist(),
                                            format_func=lambda x: next((s["Name"] for s in schema_data if s["ID"] == x), x))
            
            if selected_schema_id:
                try:
                    # Get schema details
                    schema_details = services["storage_manager"].get_schema(selected_schema_id)
                    
                    # Display schema details
                    with st.expander("Schema Details", expanded=True):
                        st.json(schema_details["schema"])
                    
                    # Schema actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Delete Schema", key="delete_schema"):
                            if services["storage_manager"].delete_schema(selected_schema_id):
                                st.success("Schema deleted successfully!")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to delete schema.")
                    
                    with col2:
                        if st.button("Export Schema", key="export_schema"):
                            schema_json = json.dumps(schema_details["schema"], indent=2)
                            st.download_button(
                                label="Download JSON",
                                data=schema_json,
                                file_name=f"{schema_details['metadata']['name']}.json",
                                mime="application/json"
                            )
                except Exception as e:
                    st.error(f"Error loading schema details: {str(e)}")
    except (AttributeError, Exception) as e:
        st.error(f"Schema management functionality is not available: {str(e)}")
        st.info("Please restart the application to enable schema management.")
        schemas = []

# Tab 1: Document Processing (continued)
with tab1:
    # File upload section
    st.subheader("Document Input")
    upload_type = st.radio(
        "Select input method:",
        options=["Upload Files", "Specify Folder Path"],
        index=0,
        help="Choose to upload files directly or specify a folder containing .txt files"
    )

    batch_mode = True  # Default to batch mode for better handling
    uploaded_files = None
    folder_path = None
    
    if upload_type == "Upload Files":
        # Allow multiple file uploads
        uploaded_files = st.file_uploader(
            "Upload one or more text files", 
            type=["txt"], 
            accept_multiple_files=True,
            help="You can select multiple .txt files by holding Ctrl or Shift while selecting"
        )
        
        # If only one file is uploaded, display it differently
        if uploaded_files and len(uploaded_files) == 1:
            st.info(f"Processing single file: {uploaded_files[0].name}")
            batch_mode = False
        elif uploaded_files and len(uploaded_files) > 1:
            st.info(f"Batch processing {len(uploaded_files)} files")
            # Show list of files to be processed
            with st.expander("Files to process"):
                for i, file in enumerate(uploaded_files):
                    st.text(f"{i+1}. {file.name} ({file.size/1024:.1f} KB)")
    else:
        # Folder path input
        folder_path = st.text_input(
            "Enter folder path containing .txt files",
            placeholder="e.g., C:\\Users\\username\\Documents\\text_files",
            help="Specify the full path to a folder containing .txt files for processing"
        )
        
        # Validate folder path if provided
        if folder_path:
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # Count .txt files in the folder
                txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
                if txt_files:
                    st.success(f"Found {len(txt_files)} .txt files in the specified folder")
                    # Show list of files to be processed
                    with st.expander("Files to process"):
                        for i, file in enumerate(txt_files):
                            file_path = os.path.join(folder_path, file)
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            st.text(f"{i+1}. {file} ({file_size:.1f} KB)")
                else:
                    st.warning("No .txt files found in the specified folder")
            else:
                st.error("Invalid folder path or folder does not exist")
    
    # Output settings
    st.subheader("Output Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        # Custom output directory
        custom_output_dir = st.text_input(
            "Output Directory",
            placeholder="Leave empty for default location",
            help="Specify where to save the extraction results"
        )
    
    with col2:
        # Combine results option
        combine_results = st.checkbox(
            "Combine all results", 
            value=True,
            help="Combine results from all files into a single output file"
        )
    
    # Checkpoint functionality
    enable_checkpoints = st.checkbox(
        "Enable checkpoint recovery", 
        value=True,
        help="If processing is interrupted, resume from the last successfully processed file"
    )
    
    # Processing settings
    st.subheader("Processing Settings")
    col1, col2 = st.columns(2)
    
    with st.expander("Batch Processing Options", expanded=False):
        max_concurrent = st.slider(
            "Maximum Concurrent Files", 
            min_value=1, 
            max_value=10, 
            value=4,
            help="Maximum number of files to process concurrently. Higher values may process faster but use more resources."
        )
        
        enable_checkpoints = st.checkbox(
            "Enable Checkpoints", 
            value=True,
            help="Save progress after each file is processed. If the process is interrupted, it can be resumed from the last successful file."
        )
        
        combine_results = st.checkbox(
            "Combine Results", 
            value=True,
            help="Combine all extracted data into a single Excel file after processing."
        )
        
        cleanup_individual_files = st.checkbox(
            "Remove Individual Files After Combining", 
            value=False,
            help="Delete individual processed files after they have been combined into a single file. This saves disk space but individual results cannot be recovered."
        )
    
    # Processing priority (for backward compatibility)
    processing_priority = "Balanced"  # Default value

# Main workflow
# Function to create checkpoint file
def save_checkpoint(checkpoint_dir, processed_files):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
    with open(checkpoint_file, "w") as f:
        json.dump({
            "processed_files": processed_files,
            "timestamp": datetime.now().isoformat()
        }, f)
    return checkpoint_file

# Function to load checkpoint file
def load_checkpoint(checkpoint_dir):
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {"processed_files": [], "timestamp": None}

# Determine if we should process based on input method
should_process = False
files_to_process = []

# For direct file uploads
if upload_type == "Upload Files" and uploaded_files:
    should_process = True
    if len(uploaded_files) == 1:
        # Single file mode
        batch_mode = False
        # Display file info
        file_details = {
            "Filename": uploaded_files[0].name,
            "File size": f"{uploaded_files[0].size / 1024:.2f} KB"
        }
        st.write("File Details:", file_details)
        
        # Preview document
        text_content = uploaded_files[0].getvalue().decode("utf-8")
        with st.expander("Document Preview"):
            st.text_area("Document Content", text_content[:3000] + "..." if len(text_content) > 3000 else text_content, height=300)
    else:
        # Multiple files mode
        batch_mode = True
        files_to_process = uploaded_files

# For folder path input
elif upload_type == "Specify Folder Path" and folder_path and os.path.exists(folder_path) and os.path.isdir(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.txt')]
    if txt_files:
        should_process = True
        batch_mode = True
        files_to_process = [os.path.join(folder_path, f) for f in txt_files]

# Setup extraction parameters
extraction_params = {
    "temperature": temperature,
    "max_tokens": max_tokens,
    "document_type": doc_type,
    "schema_id": selected_schema_id
}

# Add processing priority settings
if processing_priority == "Speed":
    extraction_params["max_tokens"] = min(extraction_params["max_tokens"], 2048)  # Limit token count for speed
elif processing_priority == "Accuracy":
    extraction_params["temperature"] = min(extraction_params["temperature"], 0.3)  # Lower temperature for accuracy

# Process button for extraction
if process_button and should_process:
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(services["storage_manager"].base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Handle single file processing
    if not batch_mode and len(uploaded_files) == 1:
        with st.spinner("Processing single document..."):
            # Get the file content
            text_content = uploaded_files[0].getvalue().decode("utf-8")
            file_name = uploaded_files[0].name
            
            # Save document
            doc_id = services["storage_manager"].save_document(file_name, text_content, doc_type)
            
            # Create smart chunks
            model_ctx = selected_model.get('context_window', chunk_size)
            split_mode = 'line' if '\t' in text_content or '\u0437\u0432\u0430\u043d\u0438\u0435' in text_content.lower() else 'paragraph'
            chunks = services["context_manager"].chunk_document(
                text_content,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                context_window=model_ctx,
                split_mode=split_mode,
                advanced=True,
                model_name=selected_model.get('model_id'),
                metadata={"filename": file_name, "doc_type": doc_type}
            )
            
            st.info(f"Advanced chunking: {len(chunks)} chunks created (context_window={model_ctx}, split_mode={split_mode})")
            
            # Process the document
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            # First pass - document understanding
            status_text.text("Analyzing document structure...")
            understanding_prompt = services["prompt_manager"].create_understanding_prompt(
                text_content[:min(len(text_content), 10000)], doc_type
            )
            
            understanding_result = services["llm_service"].call_llm(
                provider=provider,
                model=selected_model,
                prompt=understanding_prompt,
                params=extraction_params,
                api_key=api_key
            )
            
            # Use understanding to generate extraction strategy
            status_text.text("Developing extraction strategy...")
            strategy = services["prompt_manager"].create_extraction_strategy(understanding_result)
            
            # Process each chunk
            all_results = []
            initial_results = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_progress = (i / total_chunks) * 0.5  # First pass is 50% of progress
                progress_bar.progress(chunk_progress)
                status_text.text(f"Processing chunk {i+1}/{total_chunks} (first pass)...")
                
                # Use optimized prompt if available, else custom, else default
                if optimized_prompt:
                    extraction_prompt = optimized_prompt.replace("{chunk}", chunk)
                elif custom_prompt.strip():
                    extraction_prompt = custom_prompt.replace("{chunk}", chunk)
                else:
                    # If a schema is selected, include it in the extraction prompt
                    if selected_schema:
                        # Create schema-guided extraction prompt
                        schema_json = json.dumps(selected_schema, indent=2)
                        extraction_prompt = {
                            "system": "You are a data extraction specialist with exceptional attention to detail. Extract data according to the provided schema.",
                            "user": f"Extract data from the following text according to the provided schema.\n\nSCHEMA:\n{schema_json}\n\nTEXT:\n{chunk}\n\n{strategy}\n\nReturn ONLY valid JSON that matches the schema structure. Do not include any explanations or markdown formatting."
                        }
                    else:
                        # Use standard extraction prompt
                        extraction_prompt = services["prompt_manager"].create_extraction_prompt(
                            chunk, doc_type, strategy
                        )
                
                # Extract data from chunk
                chunk_result = services["llm_service"].call_llm(
                    provider=provider,
                    model=selected_model,
                    prompt=extraction_prompt,
                    params=extraction_params,
                    api_key=api_key
                )
                
                # Process results (existing code)
                try:
                    extracted_data = services["llm_service"].extract_json_from_text(chunk_result)
                    if extracted_data:
                        if isinstance(extracted_data, list):
                            initial_results.extend(extracted_data)
                        else:
                            if 'records' in extracted_data and isinstance(extracted_data['records'], list):
                                initial_results.extend(extracted_data['records'])
                            elif 'items' in extracted_data and isinstance(extracted_data['items'], list):
                                initial_results.extend(extracted_data['items'])
                            elif 'data' in extracted_data and isinstance(extracted_data['data'], list):
                                initial_results.extend(extracted_data['data'])
                            else:
                                initial_results.append(extracted_data)
                except Exception as e:
                    st.error(f"Failed to parse results from chunk {i+1}. Error: {str(e)}")
            
            # Save results
            if initial_results:
                results_id = services["storage_manager"].save_results(doc_id, initial_results)
                
                # Display results
                with results_container:
                    st.subheader("Extraction Results")
                    
                    # Convert to DataFrame for display
                    if initial_results:
                        try:
                            df = pd.DataFrame(initial_results)
                            st.dataframe(df)
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    csv,
                                    file_name=f"{file_name.split('.')[0]}_results.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                excel_buffer = io.BytesIO()
                                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                                excel_data = excel_buffer.getvalue()
                                st.download_button(
                                    "Download Excel",
                                    excel_data,
                                    file_name=f"{file_name.split('.')[0]}_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        except Exception as e:
                            st.error(f"Error converting results to DataFrame: {str(e)}")
                            st.json(initial_results)
                    else:
                        st.warning("No structured data was extracted from the document.")
            else:
                st.warning("No data could be extracted from the document.")
    
    # Handle batch processing (multiple files)
    elif batch_mode and (len(files_to_process) > 0):
        st.subheader("Batch Processing")
        
        # Create a progress container
        progress_container = st.container()
        with progress_container:
            batch_progress = st.progress(0)
            batch_status = st.empty()
            file_progress = st.progress(0)
            file_status = st.empty()
            results_summary = st.container()
        
        # Load checkpoint if enabled
        processed_files = []
        if enable_checkpoints:
            checkpoint_data = load_checkpoint(checkpoint_dir)
            processed_files = checkpoint_data.get("processed_files", [])
            if processed_files:
                st.info(f"Resuming from checkpoint: {len(processed_files)} files already processed")
        
        # Setup output directory
        output_dir = custom_output_dir if custom_output_dir else os.path.join(services["storage_manager"].base_dir, "batch_results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process files
        all_batch_results = []
        files_to_skip = set(processed_files)
        
        # Count files to process (excluding already processed ones)
        total_files = len(files_to_process)
        files_remaining = total_files - len(files_to_skip)
        
        batch_status.text(f"Processing {files_remaining} files out of {total_files} total files")
        
        # Process each file
        for file_index, file_item in enumerate(files_to_process):
            # Update batch progress
            batch_progress.progress((file_index) / total_files)
            
            # Get file info
            if upload_type == "Upload Files":
                # File from direct upload
                file_name = file_item.name
                file_id = file_name  # Use filename as ID for checkpoint
                
                # Skip if already processed
                if file_id in files_to_skip:
                    continue
                
                # Read file content
                try:
                    text_content = file_item.getvalue().decode("utf-8")
                except Exception as e:
                    batch_status.error(f"Error reading file {file_name}: {str(e)}")
                    continue
            else:
                # File from folder path
                file_path = file_item
                file_name = os.path.basename(file_path)
                file_id = file_path  # Use full path as ID for checkpoint
                
                # Skip if already processed
                if file_id in files_to_skip:
                    continue
                
                # Read file content
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                except Exception as e:
                    batch_status.error(f"Error reading file {file_name}: {str(e)}")
                    continue
            
            # Update status
            file_status.text(f"Processing file {file_index+1}/{total_files}: {file_name}")
            
            try:
                # Save document
                doc_id = services["storage_manager"].save_document(file_name, text_content, doc_type)
                
                # Create smart chunks
                model_ctx = selected_model.get('context_window', chunk_size)
                split_mode = 'line' if '\t' in text_content or '\u0437\u0432\u0430\u043d\u0438\u0435' in text_content.lower() else 'paragraph'
                chunks = services["context_manager"].chunk_document(
                    text_content,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                    context_window=model_ctx,
                    split_mode=split_mode,
                    advanced=True,
                    model_name=selected_model.get('model_id'),
                    metadata={"filename": file_name, "doc_type": doc_type}
                )
                
                # First pass - document understanding
                understanding_prompt = services["prompt_manager"].create_understanding_prompt(
                    text_content[:min(len(text_content), 10000)], doc_type
                )
                
                understanding_result = services["llm_service"].call_llm(
                    provider=provider,
                    model=selected_model,
                    prompt=understanding_prompt,
                    params=extraction_params,
                    api_key=api_key
                )
                
                # Use understanding to generate extraction strategy
                strategy = services["prompt_manager"].create_extraction_strategy(understanding_result)
                
                # Process each chunk
                file_results = []
                total_chunks = len(chunks)
                
                for i, chunk in enumerate(chunks):
                    # Update file progress
                    file_progress.progress((i) / total_chunks)
                    
                    # Create extraction prompt
                    if selected_schema:
                        # Create schema-guided extraction prompt
                        schema_json = json.dumps(selected_schema, indent=2)
                        extraction_prompt = {
                            "system": "You are a data extraction specialist with exceptional attention to detail. Extract data according to the provided schema.",
                            "user": f"Extract data from the following text according to the provided schema.\n\nSCHEMA:\n{schema_json}\n\nTEXT:\n{chunk}\n\n{strategy}\n\nReturn ONLY valid JSON that matches the schema structure. Do not include any explanations or markdown formatting."
                        }
                    else:
                        # Use standard extraction prompt
                        extraction_prompt = services["prompt_manager"].create_extraction_prompt(
                            chunk, doc_type, strategy
                        )
                    
                    # Extract data from chunk
                    chunk_result = services["llm_service"].call_llm(
                        provider=provider,
                        model=selected_model,
                        prompt=extraction_prompt,
                        params=extraction_params,
                        api_key=api_key
                    )
                    
                    # Process results
                    try:
                        extracted_data = services["llm_service"].extract_json_from_text(chunk_result)
                        if extracted_data:
                            if isinstance(extracted_data, list):
                                file_results.extend(extracted_data)
                            else:
                                if 'records' in extracted_data and isinstance(extracted_data['records'], list):
                                    file_results.extend(extracted_data['records'])
                                elif 'items' in extracted_data and isinstance(extracted_data['items'], list):
                                    file_results.extend(extracted_data['items'])
                                elif 'data' in extracted_data and isinstance(extracted_data['data'], list):
                                    file_results.extend(extracted_data['data'])
                                else:
                                    file_results.append(extracted_data)
                    except Exception as e:
                        file_status.warning(f"Failed to parse results from chunk {i+1} in file {file_name}")
                
                # Reset file progress
                file_progress.progress(1.0)
                
                # Save file results
                if file_results:
                    # Save to storage manager
                    results_id = services["storage_manager"].save_results(doc_id, file_results)
                    
                    # Save to output directory
                    try:
                        # Convert to DataFrame
                        df = pd.DataFrame(file_results)
                        
                        # Save as Excel
                        excel_path = os.path.join(output_dir, f"{file_name.split('.')[0]}_results.xlsx")
                        df.to_excel(excel_path, index=False, engine='openpyxl')
                        
                        # Add to combined results if enabled
                        if combine_results:
                            # Add filename column to identify source
                            df['source_file'] = file_name
                            all_batch_results.append(df)
                    except Exception as e:
                        batch_status.error(f"Error saving results for {file_name}: {str(e)}")
                
                # Mark file as processed for checkpoint
                processed_files.append(file_id)
                if enable_checkpoints:
                    save_checkpoint(checkpoint_dir, processed_files)
                
                # Update the UI with current progress
                current_progress = (len(processed_files) / total_files)
                batch_progress.progress(current_progress)
                batch_status.text(f"Processed {len(processed_files)} of {total_files} files ({int(current_progress * 100)}%)")
                
                # Force UI refresh by adding a small sleep
                time.sleep(0.1)
                # Update UI without full rerun
                st.session_state.update_counter = st.session_state.get('update_counter', 0) + 1
                
            except Exception as e:
                batch_status.error(f"Error processing file {file_name}: {str(e)}")
        
        # Complete batch progress
        batch_progress.progress(1.0)
        file_progress.progress(1.0)
        batch_status.success(f"Batch processing complete: {len(processed_files)} files processed")
        
        # Combine and save results if enabled
        if combine_results and all_batch_results:
            try:
                # Combine all DataFrames
                combined_df = pd.concat(all_batch_results, ignore_index=True)
                
                # Save combined results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_path = os.path.join(output_dir, f"combined_results_{timestamp}.xlsx")
                combined_df.to_excel(combined_path, index=False, engine='openpyxl')
                
                # Show download link
                with results_summary:
                    st.success(f"Combined results saved to: {combined_path}")
                    
                    # Create download button for combined results
                    with open(combined_path, "rb") as f:
                        combined_data = f.read()
                        st.download_button(
                            "Download Combined Results",
                            combined_data,
                            file_name=f"combined_results_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Show preview of combined results
                    st.subheader("Combined Results Preview")
                    st.dataframe(combined_df.head(100))
                    
                    # Cleanup individual files if enabled
                    if cleanup_individual_files:
                        with st.spinner("Removing individual result files..."):
                            cleanup_count = 0
                            # Get all result directories
                            result_dirs = [d for d in os.listdir(services["storage_manager"].results_dir) 
                                         if os.path.isdir(os.path.join(services["storage_manager"].results_dir, d))]
                            
                            for result_dir_name in result_dirs:
                                try:
                                    # Skip directories that don't look like UUIDs (schemas, etc.)
                                    if len(result_dir_name) != 36 or result_dir_name.count('-') != 4:
                                        continue
                                        
                                    # Full path to the result directory
                                    result_dir = os.path.join(services["storage_manager"].results_dir, result_dir_name)
                                    
                                    # Check if this directory has an Excel or CSV file that was just processed
                                    has_processed_file = False
                                    for file_path in glob.glob(os.path.join(result_dir, "*.csv")) + glob.glob(os.path.join(result_dir, "*.xlsx")):
                                        # If we find a file, mark this directory for cleanup
                                        has_processed_file = True
                                        break
                                    
                                    if has_processed_file:
                                        # Remove all files in the directory
                                        for file_path in glob.glob(os.path.join(result_dir, "*.*")):
                                            os.remove(file_path)
                                        # Remove the directory itself
                                        os.rmdir(result_dir)
                                        cleanup_count += 1
                                except Exception as e:
                                    st.warning(f"Could not remove results directory {result_dir_name}: {str(e)}")
                            
                            st.success(f"Removed {cleanup_count} individual result directories to save disk space.")
                            st.info("The combined results file contains all the extracted data.")
                    
            except Exception as e:
                with results_summary:
                    st.error(f"Error combining results: {str(e)}")

else:
    # Display instructions or sample documents
    st.info("Upload a text document to begin extraction.")
    st.markdown("""
    ### Sample Documents
    You can test the system with sample documents:
    - [Download Sample Financial Report](data/sample_financial.txt)
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Alibaba's Qwen LLM") 