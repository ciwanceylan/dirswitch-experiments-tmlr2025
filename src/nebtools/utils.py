import errno
import os
from datetime import datetime
import git


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


NEB_ROOT = get_git_root(os.path.abspath(__file__))
NEB_DATAROOT = os.path.join(NEB_ROOT, "data")
NEB_DATA_STRUCT_FEATURES_ROOT = os.path.join(NEB_ROOT, "data_struct_features")


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def check_save_dir_exists(out_file, fix=True):
    if not os.path.exists(os.path.dirname(out_file)):
        if fix:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        else:
            raise OSError("Directory of " + out_file + " does not exist.")


def make_result_folder(
    resultdir: str, experiment: str, dataset: str, methods_str: str, file_name: str
):
    folder = os.path.join(
        resultdir,
        experiment,
        dataset,
        methods_str,
        datetime.now().strftime("%Y%m%d-%H%M-%S-%f"),
    )
    results_path = os.path.join(folder, file_name)
    check_save_dir_exists(results_path)
    return results_path
