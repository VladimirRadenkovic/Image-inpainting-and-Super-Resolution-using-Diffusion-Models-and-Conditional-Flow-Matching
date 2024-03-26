from pathlib import Path


def get_latest_version(parent_directory) -> int:
    p = Path(parent_directory)
    if not p.is_dir():
        return "The specified path is not a directory"

    version_dirs = [str(child) for child in p.iterdir() if child.is_dir() and child.name.startswith("version")]
    if len(version_dirs) == 0:
        return -1
    else:
        versions = [int(dir_.split("_")[-1]) for dir_ in version_dirs]
        return max(versions)
