# prompt-injection-interp

### Hofvarpnir instructions
Launch a devbox on Hofvarpnir with the following command:
```
ctl devbox run \
    --name tony-pii \
    --container tonytwang/pii:main \
    --shared-host-dir /home/twang/code/prompt-injection-interp \
    --shared-host-dir-mount /prompt-injection-interp \
    --volume-name pii-data \
    --volume-mount /pii-data \
    --gpu 1
```
