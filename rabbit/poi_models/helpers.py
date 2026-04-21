from rabbit import common

# dictionary with class name and the corresponding filename where it is defined
baseline_models = {
    "Ones": "poi_model",
    "Mu": "poi_model",
    "Mixture": "poi_model",
    "AxisNormModel": "poi_model",
    "AxisExpModel": "poi_model",
}


def load_model(class_name, indata, *args, **kwargs):
    model = common.load_class_from_module(
        class_name, baseline_models, base_dir="rabbit.poi_models"
    )
    return model.parse_args(indata, *args, **kwargs)
