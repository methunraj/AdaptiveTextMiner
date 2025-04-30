# Advanced LLM Text Extraction System

This project provides a powerful and flexible system for extracting structured data from text documents using various Large Language Models (LLMs). It features a user-friendly Streamlit interface, support for multiple LLM providers, intelligent document chunking, schema management, batch processing with checkpointing, and an adaptive feedback loop for prompt refinement.

## Features

*   **Multi-Provider & Model Support:** Configure and switch between different LLM providers (e.g., OpenAI, Google, Anthropic, Alibaba Qwen) and models via `models.json`.
*   **Flexible Input:** Process single `.txt` files, multiple `.txt` files simultaneously, or entire folders containing `.txt` files.
*   **Intelligent Chunking:** Automatically splits large documents into manageable chunks, considering context windows and attempting to preserve semantic boundaries (paragraphs/lines).
*   **Schema Management:** Define, save, view, and delete custom JSON schemas to enforce a specific structure for extracted data.
*   **Schema-Guided Extraction:** Use saved schemas to guide the LLM during extraction, improving consistency and relevance.
*   **Batch Processing:** Efficiently process large numbers of documents.
*   **Checkpointing:** Automatically saves progress during batch processing, allowing resumption from the last successfully processed file in case of interruption.
*   **Combined Output:** Option to concatenate results from all files in a batch into a single downloadable Excel file.
*   **Adaptive Feedback Loop (Conceptual):** Includes modules designed to analyze initial extraction results and automatically refine prompts for improved accuracy on subsequent attempts (*Integration into the main app workflow is pending*).
*   **Usage & Cost Tracking:** Monitors token usage and estimates API costs based on provider pricing in `models.json`.
*   **Streamlit UI:** Interactive web interface for easy configuration, file handling, and results viewing.

## Architecture

The system is built with Python and Streamlit, comprising several key components:

*   **`app.py`:** The main Streamlit application file, handling the UI, user interactions, and orchestrating the workflow.
*   **`src/` Modules:**
    *   `multi_llm_service.py`: Manages interactions with different LLM APIs, handling API keys and provider-specific logic.
    *   `storage_manager.py`: Handles saving and loading documents, results, schemas, and feedback data to the local filesystem (`data/` directory).
    *   `prompt_manager.py`: Manages prompt templates for different tasks (understanding, extraction) and strategies.
    *   `context_manager.py`: Implements document chunking logic.
    *   `feedback_system.py`: Contains logic for analyzing extraction quality and generating adaptive prompts (currently requires integration into `app.py`).
    *   `model_registry.py`: Loads and provides access to model configurations from `models.json`.
*   **`models.json`:** Configuration file defining available LLM providers, models, capabilities, and pricing.
*   **`data/`:** Default directory for storing uploaded documents, extraction results, schemas, checkpoints, etc.
*   **`.env`:** File for storing API keys and other sensitive configuration (not tracked by Git).
*   **`requirements.txt`:** Lists Python package dependencies.

## Setup & Installation

1.  **Prerequisites:**
    *   Python 3.8 or higher.
    *   Git (optional, for cloning).

2.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    *   Create a file named `.env` in the project root directory.
    *   Add your API keys for the LLM providers you intend to use. The required environment variable name is typically `<PROVIDER_NAME>_API_KEY` (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `QWEN_API_KEY`). The application sidebar will show the expected variable name based on the selected provider.
    *   Example `.env` content:
        ```
        OPENAI_API_KEY="your_openai_api_key_here"
        QWEN_API_KEY="your_qwen_api_key_here"
        # Add other keys as needed
        ```

6.  **Model Configuration:**
    *   Review and update `models.json` if necessary to add or modify supported LLM providers and models.

## Usage

1.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

2.  **Configure (Sidebar):**
    *   Select the **Provider** (e.g., Qwen, OpenAI).
    *   Select the desired **Model** from the chosen provider.
    *   Ensure the corresponding **API Key** is loaded (from `.env`) or enter it manually.
    *   Select the **Document Type**.
    *   Optionally, choose an existing **Extraction Schema** to guide the process.
    *   Adjust **Advanced Options** (Temperature, Chunking, etc.) if needed.
    *   Optionally, provide a **Custom Extraction Prompt**.

3.  **Manage Schemas (Schema Management Tab):**
    *   **Create:** Fill the form to define fields (name, type, description) for a new schema and save it.
    *   **View/Manage:** Select an existing schema from the list to view its JSON details, export it, or delete it.

4.  **Process Documents (Document Processing Tab):**
    *   **Select Input:** Choose either "Upload Files" or "Specify Folder Path".
    *   **Provide Input:** Upload one or more `.txt` files or enter the path to a folder containing `.txt` files.
    *   **Output Settings:** Optionally specify an output directory and choose whether to combine batch results.
    *   **Checkpoints:** Enable/disable checkpointing for batch jobs.
    *   **Click "Process Document".**

5.  **View Results:**
    *   For single files, results appear directly in the main area.
    *   For batch files, progress is shown, and individual/combined results can be downloaded from the specified output directory or via download buttons after processing completes.
    *   The Usage & Cost Dashboard provides insights into API usage.

## Workflows

*(These flowcharts use Mermaid syntax. They should render correctly on platforms like GitHub.)*

### Overall Application Flow

```mermaid
graph TD
    A[Start app.py] --> B{Load Config/Models};
    B --> C[Initialize Services];
    C --> D[Display UI (Streamlit)];
    D --> E{User Interaction};
    E --> F[Configure Settings (Sidebar)];
    E --> G[Manage Schemas (Tab 2)];
    E --> H[Process Documents (Tab 1)];
    F --> E;
    G --> E;
    H --> I{Start Processing?};
    I -- Yes --> J[Run Extraction Workflow];
    I -- No --> E;
    J --> K[Display Results/Download];
    K --> E;
```

### Single Document Processing Flow

```mermaid
graph TD
    A[Process Button Clicked] --> B[Get Single File Content];
    B --> C[Save Document (StorageManager)];
    C --> D[Chunk Document (ContextManager)];
    D --> E[Create Understanding Prompt (PromptManager)];
    E --> F[Call LLM (LLMService)];
    F --> G[Create Extraction Strategy (PromptManager)];
    G --> H{Loop Through Chunks};
    H -- Next Chunk --> I[Create Extraction Prompt (Schema/Custom/Default)];
    I --> J[Call LLM (LLMService)];
    J --> K[Parse Result];
    K --> L[Append to Initial Results];
    L --> H;
    H -- All Chunks Done --> M[Save Results (StorageManager)];
    M --> N[Display Results / Provide Download];
    N --> O[End Processing];

    %% Optional Feedback Loop (Conceptual - Needs Integration) 
    % L --> FB1[Analyze Initial Results (FeedbackSystem)];
    % FB1 --> FB2[Generate Adaptive Prompt (FeedbackSystem)];
    % FB2 --> FB_H{Loop Through Chunks (Refined)};
    % FB_H -- Next Chunk --> FB_I[Use Refined Prompt];
    % FB_I --> FB_J[Call LLM];
    % FB_J --> FB_K[Parse Result];
    % FB_K --> FB_L[Append to Final Results];
    % FB_L --> FB_H;
    % FB_H -- All Chunks Done --> M; 
```

### Batch Document Processing Flow

```mermaid
graph TD
    A[Process Button Clicked] --> B[Get File List (Upload/Folder)];
    B --> C{Checkpoints Enabled?};
    C -- Yes --> D[Load Checkpoint (Processed Files)];
    C -- No --> E[Initialize Empty Processed List];
    D --> E;
    E --> F{Loop Through Files};
    F -- Next File --> G{File Already Processed?};
    G -- Yes --> F;
    G -- No --> H[Read File Content];
    H --> I[Save Document (StorageManager)];
    I --> J[Chunk Document (ContextManager)];
    J --> K[Run Understanding Pass (LLM)];
    K --> L[Create Extraction Strategy];
    L --> M{Loop Through Chunks};
    M -- Next Chunk --> N[Create Extraction Prompt (Schema/Default)];
    N --> O[Call LLM (LLMService)];
    O --> P[Parse Result];
    P --> Q[Append to File Results];
    Q --> M;
    M -- All Chunks Done --> R[Save Individual File Results (Excel)];
    R --> S{Combine Results Enabled?};
    S -- Yes --> T[Add File Results to Batch List];
    S -- No --> U[Update Checkpoint];
    T --> U;
    U --> F;
    F -- All Files Done --> V[Batch Complete Status];
    V --> W{Combine Results Enabled?};
    W -- Yes --> X[Concatenate Batch DataFrames];
    X --> Y[Save Combined Excel File];
    Y --> Z[Provide Download / Cleanup Option];
    W -- No --> AA[End Processing];
    Z --> AA;
```

### Schema Management Flow

```mermaid
graph TD
    A[Navigate to Schema Tab] --> B{View Existing Schemas?};
    B -- Yes --> C[List Schemas (StorageManager)];
    C --> D[Display Schemas in Table];
    D --> E{Select Schema?};
    E -- Yes --> F[Get Schema Details (StorageManager)];
    F --> G[Display JSON];
    G --> H{Action? (Delete/Export)};
    H -- Delete --> I[Delete Schema (StorageManager)];
    H -- Export --> J[Provide Download Button];
    I --> C; % Refresh list
    J --> E;
    E -- No --> K{Create New Schema?};
    
    B -- No --> K;
    K -- Yes --> L[Display Creation Form];
    L --> M[User Enters Details + JSON];
    M --> N[Submit Form];
    N --> O[Validate Input];
    O -- Valid --> P[Save Schema (StorageManager)];
    O -- Invalid --> Q[Show Error Message];
    P --> C; % Refresh list
    Q --> L;
    K -- No --> A; % Stay on tab
```

## Directory Structure

```
.env                 # API Keys (Create manually, not tracked by Git)
models.json          # LLM Provider/Model configurations
app.py               # Main Streamlit application
requirements.txt     # Python dependencies
README.md            # This file

data/                # Default root for storage
|-- documents/       # Stored uploaded documents
|-- results/         # Stored extraction results (per document)
|-- schemas/         # Stored extraction schemas
|-- checkpoints/     # Stores batch processing checkpoints
|-- batch_results/   # Default output for batch results
|-- feedback/        # Stored feedback data (if feedback system used)
`-- templates/       # Stored prompt templates (if used by PromptManager)

src/
|-- __init__.py
|-- context_manager.py
|-- feedback_system.py
|-- llm_service.py
|-- model_registry.py
|-- multi_llm_service.py
|-- prompt_manager.py
`-- storage_manager.py
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.

## Future Work

*   Fully integrate the `FeedbackSystem` into the main processing loop in `app.py` to enable the adaptive prompt refinement feature.
*   Add support for more document types (e.g., PDF, DOCX) potentially using libraries like `pypdf` or `python-docx`.
*   Implement more sophisticated chunking strategies.
*   Explore asynchronous processing for batch jobs to improve throughput.
*   Enhance error handling and reporting.
