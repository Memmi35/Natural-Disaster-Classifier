#!/usr/bin/env python3
"""
Setup script for IncidentVision project.

This script sets up the project environment, downloads dependencies,
and prepares the workspace for development.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command: str, cwd: str = None) -> bool:
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=cwd,
            capture_output=True,
            text=True
        )
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False


def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    print("Setting up Python environment...")
    
    # Create virtual environment
    if not os.path.exists('venv'):
        if not run_command(f"{sys.executable} -m venv venv"):
            return False
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:  # Unix-like
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        return False
    
    print("Python environment setup completed!")
    return True


def setup_frontend():
    """Set up frontend dependencies."""
    print("Setting up frontend environment...")
    
    frontend_dir = "web/frontend"
    
    if not os.path.exists(frontend_dir):
        print(f"Frontend directory {frontend_dir} not found!")
        return False
    
    # Install Node.js dependencies
    if not run_command("npm install", cwd=frontend_dir):
        return False
    
    print("Frontend environment setup completed!")
    return True


def create_directories():
    """Create necessary project directories."""
    print("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "logs",
        "checkpoints",
        "results",
        "tests/unit",
        "tests/integration"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep files for empty directories
        gitkeep_path = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            Path(gitkeep_path).touch()
    
    print("Project directories created!")
    return True


def setup_git():
    """Initialize git repository and configure hooks."""
    print("Setting up git repository...")
    
    if not os.path.exists('.git'):
        if not run_command("git init"):
            return False
    
    # Create basic .gitignore if it doesn't exist
    if not os.path.exists('.gitignore'):
        print("Warning: .gitignore not found. Please create one.")
    
    # Set up pre-commit hook (optional)
    hooks_dir = ".git/hooks"
    if os.path.exists(hooks_dir):
        pre_commit_hook = os.path.join(hooks_dir, "pre-commit")
        if not os.path.exists(pre_commit_hook):
            with open(pre_commit_hook, 'w') as f:
                f.write("""#!/bin/sh
# Pre-commit hook for IncidentVision
echo "Running pre-commit checks..."

# Run tests
python -m pytest tests/ --quiet

# Check code formatting
black --check src/ scripts/
flake8 src/ scripts/

echo "Pre-commit checks completed!"
""")
            
            # Make hook executable (Unix-like systems)
            if os.name != 'nt':
                os.chmod(pre_commit_hook, 0o755)
    
    print("Git repository setup completed!")
    return True


def verify_installation():
    """Verify that all components are properly installed."""
    print("Verifying installation...")
    
    # Check Python packages
    try:
        import torch
        import torchvision
        import pytorch_lightning
        import fastapi
        print("✓ Python packages installed correctly")
    except ImportError as e:
        print(f"✗ Python package import failed: {e}")
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("! CUDA not available (CPU only)")
    except:
        print("! Could not check CUDA availability")
    
    # Check frontend setup
    frontend_dir = "web/frontend"
    if os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("✓ Frontend dependencies installed")
    else:
        print("✗ Frontend dependencies not found")
        return False
    
    print("Installation verification completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Setup IncidentVision project')
    parser.add_argument(
        '--skip-python',
        action='store_true',
        help='Skip Python environment setup'
    )
    parser.add_argument(
        '--skip-frontend',
        action='store_true',
        help='Skip frontend setup'
    )
    parser.add_argument(
        '--skip-git',
        action='store_true',
        help='Skip git setup'
    )
    args = parser.parse_args()
    
    print("="*60)
    print("IncidentVision Project Setup")
    print("="*60)
    
    success = True
    
    # Create directories
    if not create_directories():
        success = False
    
    # Setup Python environment
    if not args.skip_python:
        if not setup_python_environment():
            success = False
    
    # Setup frontend
    if not args.skip_frontend:
        if not setup_frontend():
            success = False
    
    # Setup git
    if not args.skip_git:
        if not setup_git():
            success = False
    
    # Verify installation
    if success:
        if not verify_installation():
            success = False
    
    print("="*60)
    if success:
        print("🎉 Project setup completed successfully!")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your API keys")
        print("2. Place your trained model in the models/ directory")
        print("3. Run: python scripts/train_model.py --config configs/resnet18_config.yaml")
        print("4. Start the web application: docker-compose up")
    else:
        print("❌ Project setup failed. Please check the errors above.")
        sys.exit(1)
    
    print("="*60)


if __name__ == '__main__':
    main()