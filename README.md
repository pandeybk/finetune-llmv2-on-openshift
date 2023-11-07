# finetune-llmv2-on-openshift

## Build and Run using podman
```
podman build -t finetune:v0.2 .

mkdir -p cache
chown -R 1001:1001 cache
podman run -e HF_TOKEN='<huggingface-token>' -v ${PWD}/cache/:/opt/app-root/src/.cache finetune:v0.2
```