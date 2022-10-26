# ViViT

**IMPORTANT!!! You need at least 8GB of VRAM to train this model!!!**

Minimal implementation of ViViT: A [Video Vision Transformer](https://arxiv.org/abs/2103.15691) by Arnab et al., a pure Transformer-based model for video classification.

This repo contains the code used to train the [Video Vision Transformer model](https://huggingface.co/keras-io/video-vision-transformer) demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/video-vision-transformer-CT).

The original implementation of the model can be found on the keras documentation [here](https://keras.io/examples/vision/vivit/).


# How to push a model to Hugging Face Hub

1. (Optional) You can use the Dockerfile to build a dev container with all the dependencies installed.

2. To push a model to the Hugging Face Hub, you need to have a Hugging Face account and to be logged in with the `huggingface-cli` or use the `HUGGINGFACE_TOKEN` environment variable.

3. In the `params.yaml` file: 
    - Change the `repo_name` to the name of the model you want to push to the Hub. 
    - Activate the feature flag `feature_flag_push_to_hub`.

4. Run the following command to download the data, train the model and push it to the Hub:

```bash
dvc repro
```

## Future work

- [ ] Split train.py into different files (data, model, training, evaluation, push_to_hub)
- [ ] Use GitHub Actions to push the model to the Hub when a commit is tagged
