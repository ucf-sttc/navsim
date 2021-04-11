import attr
import cattr
from . import ObjDict

@attr.s(auto_attribs=True)
class BaseConfig:
    def as_dict(self):
        return cattr.unstructure(self)

"""
@attr.s(auto_attribs=True)
class Config(BaseConfig):
    # Options: command line -> yaml_file -> defaults

    env_config: EnvConfig = attr.ib(factory=EnvConfig)
    run_config: EnvConfig = attr.ib(factory=EnvConfig)

    @staticmethod
    def from_args_dict(args):
        # create config dict from command line default values
        # If config file given in command line
        # if the config file doesnt exist then error and exit
        # else read config and overwrite default values
        # add command line values
        # let us load config file first
        config_file = args.config_file
        if config_file:
            try:
                conf = ObjDict().load_from_file(config_file)
            except FileNotFoundError as e:
                if args.config_file is not None:
                    abs_path = os.path.abspath(config_file)
                    raise OSError(f"Config file could not be found at {abs_path}.") from e
            if conf:
                print(f'Following configuration loaded from {config_file}')
                print(conf.to_yaml())
        else:
            print(f"Config file not specified in command line, continuing.")

        return True
"""