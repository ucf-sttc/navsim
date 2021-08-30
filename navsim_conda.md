# TODO: Singularity

## TODO: Clean up the following section

For tmux hotkeys press ctrl+b then following key

* Start tmux session: tmux new -s <session name>
* Open another tmux shell: ctrl + b, % (vertical pane) Or ctrl + b, " (horizontal pane)
* Move between panes: ctrl + <left, right, up, down>
* Detach from tmux session: ctrl + b, d  (detach from tmux session)
* Attach to existing tmux session: tmux attach -t <session name>
* Exit Session: Type exit into all open shells within session

## TODO: To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

singularity pull docker://$repo/navsim:$ver
singularity shell --nv \
-B <absolute path of sim binary folder>  not needed if path to binary is inside $HOME folder  
-B <absolute path of current folder>  not needed if path to current folder is inside $HOME folder
navsim_$ver.sif


For IST Devs: From local docker repo for development purposes:

SINGULARITY_NOHTTPS=true singularity pull docker://$repo/navsim:$ver
