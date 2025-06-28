import os
import re
import glob

def fix_imports_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Fix relative imports
    content = re.sub(r'from\s+\.([a-zA-Z0-9_]+)\s+import', r'from backend.\1 import', content)
    
    # Fix direct imports of backend modules
    backend_modules = ['api', 'auth', 'config', 'feedback_api', 'evaluation_api', 'utils', 'models']
    for module in backend_modules:
        content = re.sub(fr'from\s+{module}\s+import', fr'from backend.{module} import', content)
        content = re.sub(fr'import\s+{module}(?!\w)', fr'import backend.{module}', content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    # Get all Python files in the backend directory
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
    python_files = glob.glob(os.path.join(backend_dir, '**', '*.py'), recursive=True)
    
    # Fix imports in each file
    for file_path in python_files:
        fix_imports_in_file(file_path)
    
    print("All imports fixed!")

if __name__ == "__main__":
    main()