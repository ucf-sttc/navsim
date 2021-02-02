from typing import Optional, List

import ezai_util
import navsim
from ezai_util import ObjDict
import argparse

def parse_command_line(argv: Optional[List[str]] = None):
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "navsim_config_file",
        nargs='?',
        default = "navsim_conf.json")

    argparser.add_argument(
        "--env_path",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
    )
    argparser.add_argument(
        "--resume",
        default=False,
        dest="resume",
        action='store_true',
        help="Whether to resume the previous run",
    )

    argparser.add_argument(
        "--run_id",
        default="navsim_demo",
        dest="run_id",
        help="Run ID",
    )

    args = ObjDict(vars(argparser.parse_args(argv)))
    return args  # args is dictionary of variables

def main():
    """
    TODO: Implement configuration checks
    :return:
    """
    args = parse_command_line()
    print('arguments passed:')
    print(args)
    conf = ObjDict().load_from_file(args["navsim_config_file"])
    #conf.run_conf = DictObj(conf.run_conf)
    #conf.env_conf = DictObj(conf.env_conf)

    #override with CLI args
    if args["env_path"]:
        conf.env_info["env_path"]=args["env_path"]
    print("Configuration:")
    print(conf)
    trainer = navsim.Trainer(run_id=args["run_id"],
                             resume=args["resume"],
                             conf=conf)

    trainer.train()
    print("training finished")
    trainer.env_close()
    print("env closed")
    trainer.files_close()
    print("files closed")

# For python debugger to directly run this script
if __name__ == "__main__":
    main()