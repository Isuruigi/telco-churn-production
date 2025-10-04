import setuptools

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file for the package dependencies
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="telco_churn_prediction",
    version="0.1.0",
    author="Isuru Chathuranga", 
    author_email="isuruigic@gmail.com", 
    description="A complete ML project for Telco Churn Prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
   
    
    # The package directory is 'src'.
    package_dir={'': 'src'},
    # Automatically find all packages in the 'src' directory.
    packages=setuptools.find_packages(where='src'),
    
    # List of dependencies
    install_requires=requirements,
    
    python_requires='>=3.9',
    
    # Create command-line entry points
    entry_points={
        'console_scripts': [
            'telco-train=scripts.train_model:main',
            'telco-predict=scripts.predict:main',
            'telco-demo=scripts.demo:run_demo',
            'telco-compare=scripts.compare_models:main',
            'telco-orchestrator=scripts.pipeline_orchestrator:main'
        ],
    },
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License", 
        "Operating System :: OS Independent",
    ],
)
