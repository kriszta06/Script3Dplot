import sys
import subprocess
import os
import pkg_resources

print("="*80)
print("Python Environment Diagnostic Tool")
print("="*80)

# 1. System information
print("\n[System Information]")
print(f"Platform: {sys.platform}")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"System PATH: {sys.path}")

# 2. Check seaborn installation
print("\n[Seaborn Check]")
try:
    import seaborn
    version = seaborn.__version__
    print(f"✓ Seaborn is installed (version {version})")
except ImportError:
    print("✗ Seaborn is NOT accessible!")
    
    # Attempt to install seaborn
    print("\nAttempting to install seaborn...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        print("✓ Seaborn installation successful!")
        try:
            import seaborn
            print(f"✓ Seaborn version: {seaborn.__version__}")
        except ImportError:
            print("✗ Still unable to import seaborn after installation!")
    except Exception as e:
        print(f"✗ Failed to install seaborn: {e}")

# 3. Check other critical packages
print("\n[Critical Packages Check]")
packages = ["numpy", "matplotlib", "scikit-learn", "scipy", "pandas"]
for pkg in packages:
    try:
        dist = pkg_resources.get_distribution(pkg)
        print(f"✓ {pkg} ({dist.version})")
    except pkg_resources.DistributionNotFound:
        print(f"✗ {pkg} NOT installed")

# 4. Virtual environment check
print("\n[Virtual Environment Check]")
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✓ Running in a virtual environment")
    print(f"  Virtual environment path: {sys.prefix}")
else:
    print("✗ Not running in a virtual environment")

# 5. Package locations
print("\n[Package Locations]")
for pkg in ["seaborn"] + packages:
    try:
        dist = pkg_resources.get_distribution(pkg)
        print(f"{pkg}: {dist.location}")
    except pkg_resources.DistributionNotFound:
        print(f"{pkg}: Not found")

print("\n" + "="*80)
print("Diagnostic complete. Please share this output for further assistance.")
print("="*80)