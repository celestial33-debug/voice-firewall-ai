from huggingface_hub import HfApi

username = input("Enter your Hugging Face username: ").strip()
token = input("Paste your Hugging Face Write token (hf_...): ").strip()
repo_id = f"{username}/voice-firewall-ai"

api = HfApi(token=token)

print(f"\n1/2 Creating (or connecting to) Docker Space: {repo_id} ...")
api.create_repo(
    repo_id=repo_id,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
)

print("2/2 Uploading project files and demo_assets (skipping .venv and node_modules)...")
api.upload_folder(
    folder_path=".",
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=[
        ".venv/**",
        ".venv",
        "**/node_modules/**",
        "frontend/dist/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "demo_assets/*_raw.wav",
        "demo_assets/*.m4a",
        ".git/**",
        "deploy_hf.py",
    ],
)

subdomain = username.lower().replace("_", "-")
print("\n=== DEPLOYMENT UPLOADED ===")
print(f"Build logs:  https://huggingface.co/spaces/{repo_id}?logs=build")
print(f"Live App:    https://{subdomain}-voice-firewall-ai.hf.space")