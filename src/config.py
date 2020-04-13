import os


def check_create_path(*args):
    for i, arg in enumerate(args):
        if i > 0:
            path = os.path.join(path, arg)
        else:
            path = arg
        if not os.path.exists(path):
            os.mkdir(path)
    return path


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = check_create_path(project_dir, "results/")
data_dir = check_create_path(project_dir, "data/")
models_dir = check_create_path(project_dir, "src", "Models")
