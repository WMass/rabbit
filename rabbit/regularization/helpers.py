from rabbit import common

# dictionary with class name and the corresponding filename where it is defined
baseline_regularizations = {
    "SVD": "svd",
}


def load_regularizer(class_name, *args, **kwargs):
    regularization = common.load_class_from_module(
        class_name, baseline_regularizations, base_dir="rabbit.regularization"
    )
    return regularization(*args, **kwargs)
