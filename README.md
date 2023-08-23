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
    --gpu 1 \
    --cpu 12 \
    --memory 30Gi
```
Adjust to your own username and path to the repo as necessary.

You can check the status of your devbox with the command
```
ctl job list
kubectl view-allocations -r gpu
```
Once your devbox is running,
you can launch a vscode server on it with the command
```
ctl devbox vscode --name tony-pii
```
This command needs to be running to keep the vscode server alive,
so you should run it in a tmux session or something similar.
You may need to unregister your old vscode tunnel,
see https://github.com/microsoft/vscode-remote-release/issues/8544#issuecomment-1570741832 for details.

If you want to update the container,
run the following from the root of the repo:
```
docker build -t tonytwang/pii:main .
docker push tonytwang/pii:main
```
You can check the image here:
https://hub.docker.com/repository/docker/tonytwang/pii
