from rabbit import common

# dictionary with class name and the corresponding filename where it is defined
baseline_models = {
    "Ones": "param_model",
    "Mu": "param_model",
    "Mixture": "param_model",
    "ABCD": "abcd_model",
    "ExtendedABCD": "extended_abcd_model",
    "SmoothABCD": "smooth_abcd_model",
    "SmoothExtendedABCD": "smooth_extended_abcd_model",
    "ABCDIsoMT": "abcd_isomtmt_model",
    "ExtendedABCDIsoMT": "abcd_isomtmt_model",
    "SmoothABCDIsoMT": "abcd_isomtmt_model",
    "SmoothExtendedABCDIsoMT": "abcd_isomtmt_model",
}


def load_model(class_name, indata, *args, **kwargs):
    model = common.load_class_from_module(
        class_name, baseline_models, base_dir="rabbit.param_models"
    )
    return model.parse_args(indata, *args, **kwargs)


def load_models(model_specs, indata, **kwargs):
    """Load one or more param models and return a single model (or composite).

    Args:
        model_specs: list of lists, e.g. [["Mu"], ["ABCD", "nonprompt", "ch_A", ...]]
        indata: FitInputData instance
        **kwargs: passed to each model's parse_args (e.g. from vars(args))
    """
    from rabbit.param_models.param_model import CompositeParamModel

    models = [load_model(spec[0], indata, *spec[1:], **kwargs) for spec in model_specs]
    if len(models) == 1:
        return models[0]
    return CompositeParamModel(
        models, allowNegativeParam=kwargs.get("allowNegativeParam", False)
    )
