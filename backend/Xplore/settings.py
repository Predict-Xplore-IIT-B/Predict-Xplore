# Define where container outputs are stored
ADDITIONAL_OUTPUTS_ROOT = os.path.join(BASE_DIR, 'outputs')

# Ensure the directory exists
os.makedirs(ADDITIONAL_OUTPUTS_ROOT, exist_ok=True)