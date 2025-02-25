import os
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model_from_hf(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",  # Adjust based on quantization level you want
    local_dir="/home/decoder/ai/models/Meta-Llama-3.1-8B-Instruct-GGUF"
):
    """
    Download a model from Hugging Face Hub
    
    Args:
        repo_id (str): The Hugging Face repository ID
        filename (str): The specific file to download from the repo
        local_dir (str): Local directory to save the model
        
    Returns:
        str: Path to the downloaded model file
    """
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Full path where the model will be saved
    local_path = os.path.join(local_dir, filename)
    
    # Check if model already exists
    if os.path.exists(local_path):
        logger.info(f"Model already exists at {local_path}")
        return local_path
    
    # Download the model
    logger.info(f"Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Set to True if you want to use symlinks
        )
        logger.info(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

if __name__ == "__main__":
    # List available files in the repository
    try:
        from huggingface_hub import list_repo_files
        repo_id = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
        files = list_repo_files(repo_id)
        print("Available files in repository:")
        for file in files:
            if file.endswith(".gguf"):
                print(f" - {file}")
        print("\nYou can specify any of these filenames when calling download_model_from_hf()")
    except Exception as e:
        print(f"Couldn't list repository files: {e}")
    
    # Download the default model (Q4_K_M quantization)
    model_path = download_model_from_hf()
    print(f"\nModel downloaded to: {model_path}")
    
    # Print approximate model size
    size_bytes = os.path.getsize(model_path)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    print(f"Model size: {size_gb:.2f} GB")