import os


def generate_api_docs():
    # Directory containing the source files
    source_dir = os.path.join("source", "api")

    # Create the directory if it doesn't exist
    os.makedirs(source_dir, exist_ok=True)

    # Run sphinx-apidoc to generate the module documentation
    os.system(f"sphinx-apidoc --force --maxdepth 2 --separate --module-first --output-dir {source_dir} ../lipana")


if __name__ == "__main__":
    # Generate the API documentation
    generate_api_docs()

    # Build the documentation
    os.system("make clean")
    os.system("make html")
