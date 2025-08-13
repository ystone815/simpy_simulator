#!/usr/bin/env python3
"""
Fix import paths after restructuring project
"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix import statements in a single file"""
    print(f"Processing: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Fix base module imports
    content = re.sub(r'from base\.', 'from src.base.', content)
    content = re.sub(r'import base\.', 'import src.base.', content)
    
    # Fix components module imports - need to determine correct path based on current file location
    file_parts = Path(file_path).parts
    
    if 'src/components' in str(file_path):
        # Files in src/components/
        if 'src/components/gpu' in str(file_path):
            # Files in GPU subdirectory
            content = re.sub(r'from components\.gpu_', 'from src.components.gpu.gpu_', content)
            content = re.sub(r'from components\.streaming_multiprocessor', 'from src.components.gpu.streaming_multiprocessor', content)
            content = re.sub(r'from components\.h100_gpu', 'from src.components.gpu.h100_gpu', content)
            content = re.sub(r'from components\.b200_gpu', 'from src.components.gpu.b200_gpu', content)
            content = re.sub(r'from components\.gpu_memory_hierarchy', 'from src.components.gpu.gpu_memory_hierarchy', content)
            content = re.sub(r'from components\.nvme_doorbell_system', 'from src.components.storage.nvme_doorbell_system', content)
            content = re.sub(r'from components\.nvme_fallback_patterns', 'from src.components.storage.nvme_fallback_patterns', content)
        elif 'src/components/storage' in str(file_path):
            # Files in storage subdirectory
            content = re.sub(r'from components\.nvme_doorbell_system', 'from src.components.storage.nvme_doorbell_system', content)
        else:
            # Files in main components directory
            content = re.sub(r'from components\.nvme_driver', 'from src.components.nvme_driver', content)
            content = re.sub(r'from components\.sram', 'from src.components.sram', content)
    
    elif 'src/base' in str(file_path):
        # Files in base directory
        content = re.sub(r'from components\.gpu_warp', 'from src.components.gpu.gpu_warp', content)
        content = re.sub(r'from components\.h100_gpu', 'from src.components.gpu.h100_gpu', content)
        content = re.sub(r'from base\.packet', 'from src.base.packet', content)
    
    elif 'src/workloads' in str(file_path):
        # Files in workloads directory
        content = re.sub(r'from components\.cugraph_inspired_gnn', 'from src.workloads.gnn.cugraph_integration', content)
        content = re.sub(r'from components\.configurable_gnn_engine', 'from src.workloads.gnn.configurable_gnn_engine', content)
        content = re.sub(r'from components\.gpu_warp', 'from src.components.gpu.gpu_warp', content)
        content = re.sub(r'from components\.nvme_doorbell_system', 'from src.components.storage.nvme_doorbell_system', content)
        content = re.sub(r'from components\.nvme_fallback_patterns', 'from src.components.storage.nvme_fallback_patterns', content)
    
    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated imports in {file_path}")
        return True
    else:
        print(f"  ⚪ No changes needed in {file_path}")
        return False

def main():
    """Fix imports in all Python files"""
    print("🔧 Fixing import paths...")
    
    src_dir = Path("src")
    if not src_dir.exists():
        print("❌ src directory not found. Run from project root.")
        return
    
    python_files = list(src_dir.rglob("*.py"))
    updated_files = 0
    
    for py_file in python_files:
        if fix_imports_in_file(py_file):
            updated_files += 1
    
    print(f"\n📊 Summary:")
    print(f"  Files processed: {len(python_files)}")
    print(f"  Files updated: {updated_files}")
    print(f"  Files unchanged: {len(python_files) - updated_files}")
    
    if updated_files > 0:
        print(f"\n✅ Import paths fixed successfully!")
    else:
        print(f"\n⚪ No import path changes were needed.")

if __name__ == "__main__":
    main()