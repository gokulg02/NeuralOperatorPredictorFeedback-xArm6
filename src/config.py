import numpy as np
import tomllib

### CONFIGURATION CLASS ###
# Currently set with defaults, use load_config to load more values

class SimulationConfig:
    def __init__(self, configFilePath=None):
        if configFilePath is None:
            raise Exception("Please specify a valid sim config path")
        else:
            with open(configFilePath) as fileObj:
                config = fileObj.read()
                config = tomllib.loads(config)
            if config["dataset"]["device_name"] != "numerical":
                # ONLY NEED TORCH FOR NON NUMERICAL EXAMPLES
                import torch
                config["dataset"]["device"] = torch.device(config["dataset"]["device_name"])
            else:
                config["dataset"]["device"] = None

            self.string_dict = {
                    "dof": config["baxter"]["dof"],
                    "alpha_mat": np.identity(config["baxter"]["dof"]),
                    "beta_mat": np.identity(config["baxter"]["dof"]),
                    "T": config["baxter"]["T"],
                    "period": config["baxter"]["period"],
                    "scaling": config["baxter"]["scaling"],
                    "dt": config["baxter"]["dt"],
                    "t": np.arange(0, config["baxter"]["T"], config["baxter"]["dt"]),
                    "D": config["baxter"]["D"],
                    "nD": int(round(config["baxter"]["D"] / config["baxter"]["dt"])),
                    "num_data": config["dataset"]["num_data"],
                    "deviation": config["dataset"]["deviation"],
                    "dataset_filename":config["dataset"]["dataset_filename"],
                    "test_size": config["dataset"]["test_size"],
                    "batch_size": config["dataset"]["batch_size"],
                    "random_state": config["dataset"]["random_state"],
                    "device_name": config["dataset"]["device_name"],
                    "device": config["dataset"]["device"]}


    def update_config(self, **kwargs):
        for k, val in kwargs.items():
            try:
                self.string_dict[k] = val
            except:
                raise Exception("Model parameter not valid. Please see the template config files for example of model parameter names")

    def __getattr__(self, name):
        return self.string_dict[name]

class ModelConfig:
    def __init__(self, configFilePath=None):
        if configFilePath is None:
            raise Exception("Please specify a valid model config path")
        else:
            with open(configFilePath) as fileObj:
                config = fileObj.read()
                config = tomllib.loads(config)
            if config["model"]["device_name"] != "numerical":
                # ONLY NEED TORCH FOR NON NUMERICAL EXAMPLES
                import torch
                config["model"]["device"] = torch.device(config["model"]["device_name"])
            else:
                config["model"]["device"] = None
            
        # Used in update config.
        self.string_dict = {
                "model_filename": config["model"]["model_filename"] ,
                "model_type": config["model"]["model_type"],
                "epochs": config["model"]["epochs"],
                "gamma": config["model"]["gamma"],
                "learning_rate": config["model"]["learning_rate"],
                "weight_decay": config["model"]["weight_decay"],
                "device_name": config["model"]["device_name"],
                "device": config["model"]["device"]
                }
        match self.string_dict["model_type"]:
            case "GRU":
                self.string_dict["num_layers"] =  config["GRU"]["num_layers"]
                self.string_dict["hidden_size"] =  config["GRU"]["hidden_size"]
                self.string_dict["input_channel"] =  config["GRU"]["input_channel"]
                self.string_dict["output_channel"] =  config["GRU"]["output_channel"]
                self.string_dict["dim_x"] = None
                self.string_dict["projection_width"] = None
            case "DeepONet":
                self.string_dict["num_layers"] =  config["DeepONet"]["num_layers"]
                self.string_dict["hidden_size"] =  config["DeepONet"]["hidden_size"]
                self.string_dict["input_channel"] =  config["DeepONet"]["input_channel"]
                self.string_dict["output_channel"] =  config["DeepONet"]["output_channel"]
                self.string_dict["dim_x"] = config["DeepONet"]["dim_x"]
                self.string_dict["projection_width"] = config["DeepONet"]["projection_width"]
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
