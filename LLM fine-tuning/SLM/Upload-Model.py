from huggingface_hub import create_repo

# Create repo

HF_USERNAME = "bakarqureshi"

repo_id = f"{HF_USERNAME}/FoodExtract-gemma-3-270m-fine-tune-v1"

create_repo(repo_id=repo_id,
            repo_type="model",
            exist_ok=True, 
            private=False)