# prompt-injection-interp

### Repository setup
Clone the repository (recursively to get the submodules) and then run:
```
pip install -r requirements.txt
pip install -e .
```
The requirements.txt step should be unnecessary if you are runn from the docker container.

### AC/DC setup
This is WIP by Tony and may not work, but to setup AC/DC, run:
```
pip install -e submodules/acdc
```

### Docker container
Our official docker environment is here:
https://hub.docker.com/repository/docker/tonytwang/pii

### OpenAI API setup
To run code that interfaces with the OpenAI API,
make sure you set the
`OPENAI_API_KEY` and `OPENAI_ORG_ID`
environment variables.
