import os
import queue
from pathlib import Path, PosixPath


def get_absolute_path(directory, project_root='pytorch_tensorflow_image_ml', ignore_hidden=True, ignore_files=True):
    """
    Gets the absolute path to a directory in the project structure using depth first search.

    Args:
        directory(str or list): The name of the folder or folder directory order to look for.
        project_root: The project root name. Generally should not be changed.
        ignore_hidden: For future use, for now, throws error because it cannot handle hidden files.
        ignore_files: For future use, for now
        , throws error because it is expecting directories.

    Returns:

    """
    if not ignore_hidden:
        raise NotImplementedError('ignore_hidden is not supported currently.')
    if not ignore_files:
        raise NotImplementedError('ignore_files is not supported currently.')
    if type(directory) is not list and type(directory) is not str:
        raise TypeError(f'directory needs to be a list or a str. Currently it is a {directory}')

    directory = [directory] if type(directory) is str else directory
    full_path = Path(__file__).parents[1]  # type: PosixPath

    # Move up the path address to the project root
    while full_path.name != project_root:
        full_path = full_path.parents[1]

    # Find the path to the directory
    searched_directory = queue.Queue()
    searched_directory.put_nowait(full_path)
    # Will do a depth first search
    while not searched_directory.empty():
        full_path_str = str(searched_directory.get_nowait())
        if os.path.exists(os.path.join(full_path_str, *tuple(directory))):
            return os.path.join(full_path_str, *tuple(directory))

        for inner_dir in os.listdir(full_path_str):
            if str(inner_dir).__contains__('.'):
                continue  # Directory is either a file, or hidden. Skip this

            searched_directory.put_nowait(os.path.join(full_path_str, inner_dir))

    raise IOError(f'Path to {directory} not found.')
