import time
import argparse
from pathlib import Path
import subprocess
import os


def monitorsubprocesses(proc_list, timeout):
    while len(proc_list) != 0:
        for p in proc_list:
            status = p.poll()
            if status == 0:
                print("Success", p, len(proc_list))
                proc_list.remove(p)

            elif status != None:
                print("Failure", p, len(proc_list), status)
                proc_list.remove(p)
        time.sleep(timeout)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('binary', metavar='B', type=Path,
                        default=None, #"/mnt/AICOOP_binaries/Build2.4.4/Berlin_Walk_V2.x86_64",
                        help="File path to Unity game binary")
    # TODO find this dynamically with nvidia-smi and parse
    # nvidia-smi  --help-query-gpu
    # p=subprocess.Popen(["nvidia-smi", '--query-gpu=count', '--format=csv', '-i', 0])
    # p=subprocess.Popen(["nvidia-smi", '--query-gpu=memory.total', '--format=csv', '-i', 0])
    parser.add_argument('-x-disp', metavar='X', default="0", help="X DISPLAY port")
    parser.add_argument('--num-gpus', default=4, type=int, help="Number of available gpus")
    parser.add_argument('--max-proc-vram', default=5000, type=int, help="Maximum VRAM used by Unity Game")
    parser.add_argument('--single-gpu-vram', default=11178, type=int, help="Total amount of available vram per gpu")
    parser.add_argument('--timeout', default=30, type=int,
                        help="Amount of time in seconds between process status checks")
    parser.add_argument('--override-num-proc', type=int, help="Manually set number of processes per gpu")

    args = parser.parse_args()

    print(args)
    # "python ../benchmarks/benchmark.py  ../../AICOOP_binaries/Build2.4.4/Berlin_Walk_V2.x86_64 -a VectorVisual"
    unity_binary_path = args.binary

    gpu_env = os.environ.copy()

    # Start benchmark.py on each gpu as process
    proc_list = []
    proc_per_gpu = int(args.single_gpu_vram / args.max_proc_vram)
    if args.override_num_proc is not None:
        proc_per_gpu = args.override_num_proc

    for gpu_idx in range(0, args.num_gpus):
        for proc_idx in range(0, proc_per_gpu):
            # proc_list.append(subprocess.Popen(["ls", "-la"]))
            gpu_env["DISPLAY"] = ":" + str(args.x_disp) + "." + str(gpu_idx)
            sp_args = ["navsim-benchmark", unity_binary_path, "-a", "VectorVisual", "--worker_id",
                       str(proc_idx + (gpu_idx * proc_per_gpu))]
            print(sp_args)
            proc_list.append(subprocess.Popen(sp_args, env=gpu_env))

    # Monitor subprocess list by checking return codes on a loop
    monitorsubprocesses(proc_list, args.timeout)

    print("DONE")

# For python debugger to directly run this script
if __name__ == "__main__":
    main()

