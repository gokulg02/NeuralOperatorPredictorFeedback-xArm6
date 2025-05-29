import numpy as np
import tomllib
import pprint

### CONFIGURATION CLASS ###

class SimulationConfig:
    def __init__(self, configFilePath=None):
        if configFilePath is None:
            raise Exception("Please specify a valid sim config path")
        elif isinstance(configFilePath, str):
            with open(configFilePath) as fileObj:
                config = fileObj.read()
                config = tomllib.loads(config)
        elif isinstance(configFilePath, dict):
            config = configFilePath
        else:
            raise Exception("Config not specified. Please pass either a dictionary with config variables or proper filepath as a string")

        if config["dataset"]["device_name"] != "numerical":
            if config["dataset"]["device_name"] != "numerical":
                # ONLY NEED TORCH FOR NON NUMERICAL EXAMPLES
                import torch
                config["dataset"]["device"] = torch.device(config["dataset"]["device_name"])
            else:
                config["dataset"]["device"] = None
            
        self.string_dict = {
                "T": config["dynamics"]["T"],
                "dt": config["dynamics"]["dt"],
                "t": np.arange(0, config["dynamics"]["T"], config["dynamics"]["dt"]),
                "D": config["dynamics"]["D"],
                "nD": int(round(config["dynamics"]["D"] / config["dynamics"]["dt"])),
                "x": np.arange(0, 1, config["dynamics"]["dx"]),
                "dx": config["dynamics"]["dx"],
                "batch_size": config["dataset"]["batch_size"], 
                "device_name": config["dataset"]["device_name"], 
                "device": config["dataset"]["device"], 
                "random_state": config["dataset"]["random_state"], 
                "test_size": config["dataset"]["test_size"]
                }

        if "phi" in config["dynamics"]:
            self.string_dict["phi"] =  config["dynamics"]["phi"]
            self.string_dict["phi_inv"] = config["dynamics"]["phi_inv"]
            self.string_dict["phi_inv_deriv"] = config["dynamics"]["phi_inv_deriv"]
            self.string_dict["a"] = config["dynamics"]["a"]
            self.string_dict["b"] = config["dynamics"]["b"]

    def update_config(self, **kwargs):
        for k, val in kwargs.items():
            try:
                self.string_dict[k] = val
            except:
                raise Exception("Model parameter not valid. Please see the template config files for example of model parameter names")

    def __getattr__(self, name):
        return self.string_dict[name]

    def __str__(self):
        return pprint.pformat(self.string_dict)

class ModelConfig:
    def __init__(self, configFilePath=None):
        if configFilePath is None:
            raise Exception("Please specify a valid model config path")
        elif isinstance(configFilePath, str):
            with open(configFilePath) as fileObj:
                config = fileObj.read()
                config = tomllib.loads(config)
        elif isinstance(configFilePath, dict):
            config = configFilePath
        else:
            raise Exception("Config not specified. Please pass either a dictionary with config variables or proper filepath as a string")

        if config["train"]["device_name"] != "numerical":
            if config["train"]["device_name"] != "numerical":
                # ONLY NEED TORCH FOR NON NUMERICAL EXAMPLES
                import torch
                config["train"]["device"] = torch.device(config["train"]["device_name"])
            else:
                config["train"]["device"] = None
            
        # Used in update config.
        self.string_dict = {
                "model_filename": config["train"]["model_filename"] ,
                "model_type": config["train"]["model_type"],
                "epochs": config["train"]["epochs"],
                "gamma": config["train"]["gamma"],
                "learning_rate": config["train"]["learning_rate"],
                "weight_decay": config["train"]["weight_decay"],
                "device_name": config["train"]["device_name"],
                "device": config["train"]["device"],
                "input_channel": 0, # dummy value
                "output_channel": 0, # dummy value
                }

        match self.string_dict["model_type"]:
            case "DeepONet":
                self.string_dict["num_layers"] =  config["DeepONet"]["num_layers"]
                self.string_dict["hidden_size"] =  config["DeepONet"]["hidden_size"]
                self.string_dict["dim_x"] = config["DeepONet"]["dim_x"]
            case "FNO":
                self.string_dict["num_layers"] = config["FNO"]["num_layers"]
                self.string_dict["hidden_size"] =  config["FNO"]["hidden_size"]
                self.string_dict["modes"] =  config["FNO"]["hidden_size"]
            case "FNO+GRU":
                self.string_dict["fno_hidden_size"] =  config["FNOGRU"]["fno_hidden_size"]
                self.string_dict["gru_hidden_size"] =  config["FNOGRU"]["gru_hidden_size"]
                self.string_dict["fno_num_layers"] =  config["FNOGRU"]["fno_num_layers"]
                self.string_dict["gru_num_layers"] =  config["FNOGRU"]["gru_num_layers"]
                self.string_dict["modes"] =  config["FNOGRU"]["modes"]
            case "DeepONet+GRU":
                self.string_dict["gru_num_layers"] =  config["DeepONetGRU"]["gru_num_layers"]
                self.string_dict["deeponet_num_layers"] =  config["DeepONetGRU"]["deeponet_num_layers"]
                self.string_dict["gru_hidden_size"] =  config["DeepONetGRU"]["gru_hidden_size"]
                self.string_dict["deeponet_hidden_size"] =  config["DeepONetGRU"]["deeponet_hidden_size"]
                self.string_dict["dim_x"] = config["DeepONetGRU"]["dim_x"]
 
            case _:
                pass

    def update_config(self, **kwargs):
        for k, val in kwargs.items():
            try:
                self.string_dict[k] = val
            except:
                raise Exception("Model parameter not valid. Please see the template config files for example of model parameter names")

    def __getattr__(self, name):
        return self.string_dict[name]

    def __str__(self):
        return pprint.pformat(self.string_dict)


