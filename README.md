# prompt-injection-interp

### Hofvarpnir instructions
Launch a devbox on Hofvarpnir with the following command:
```
ctl devbox run \
    --name tony-pii \
    --container tonytwang/pii:main \
    --shared-host-dir /home/twang/code/prompt-injection-interp \
    --shared-host-dir-mount /root/prompt-injection-interp \
    --volume-name pii-data \
    --volume-mount /pii-data \
    --gpu 1
```
You can check the status of your devbox with the command
```
ctl job list
```
Once your devbox is running,
you can launch a vscode server on it with the command
```
ctl devbox vscode --name tony-pii
```
This command needs to be running to keep the vscode server alive,
so you should run it in a tmux session or something similar.
