import os # to provide functions such as getting current directory
import shutil # to copy and removal of files

from pathlib import Path  # to work with file and directory paths (read and write files)


# Deletes all the contents (recursively) inside the given directory
def delete_contents_of_directory(dir_path: Path) -> None:
    """Deletes all the contents (recursively) inside the given directory."""
    if not os.path.isdir(dir_path):
        return
    for file_path in dir_path.iterdir():
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Creates (recursively) a directory in the specified path
def create_directory_if_not_exists(dir_path: Path) -> None:
    """Creates (recursively) a directory in the specified path."""
    os.makedirs(dir_path, exist_ok=True)

# to create directory path or empty the exist directory path
def force_empty_directory(dir_path: Path) -> None:
    """Given a directory path, it creates it if it doesn't exist. If it exists,
        it empties it."""
    if not os.path.isdir(dir_path):
        create_directory_if_not_exists(dir_path)
    else:
        delete_contents_of_directory(dir_path)
