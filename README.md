# Image captioning
As an example of how to apply transformers in practice I wanted to make a few examples showcasing different tasks. This task being the captioning of images, which is quite easy to do as there are a lot of pretrained models available. As I also wanted to show the finetuning process a bit, I will also make an example where the model predicts the used prompt to recreate the image that is shown, with a stable diffusion model.

## What are transformers?
Not sure what transformers are or how they work? No worries I've got you covered with [this repository.](https://github.com/VerleysenNiels/transformers-pytorch)
Definitely also read the recommended papers to get a better understanding.

## A good standard practice
As you know transformers require a huge amount of data in order to attain a good performance. In practice, this can be quite a show-stopper. Huge datasets are expensive and hard to come by. And even then, there is a steep cost for training your model on this huge amount of data. 

It is therefore advised to start from a pretrained model on the same task (ideally on a similar dataset) and then finetune it on your own dataset. This lets you attain a much higher performance on your dataset that doesn't have to be huge, while consuming not as many resources. But, doesn't it take quite some time and effort to find, download and use a pretrained model? Not at all! Thanks to the platform [Hugging Face.](https://huggingface.co/) This platform provides you with everything you need, from pretrained models to datasets, documentation and even courses.

To find your model you go to the models tab, select the task for which you want a model. Maybe add some more tags, like the language you are working in or the deep learning framework you want to use. Once you have found a model, you can use their transformers library in python to easily load in the model. This final part is what these example projects are for.

## Environment
I have added the conda environment yaml to the repository, as well as a Dockerfile if you want to containerize this model. The most important two libraries are of course [Transformers](https://pypi.org/project/transformers/) and [PyTorch](https://pytorch.org/) with CUDA. I am running everything locally or on a GPU server in python, but you can always copy the code to a notebook and run it for instance in [Google Colab](https://colab.research.google.com/) if you don't have access to a GPU. 

## Dataset and model
For this specific repository I am using the [diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) dataset that I found on the HuggingFace platform.

For the model I settled with a [ViT-GPT-2 model](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) which was pretrained for general purpose image captioning.

## Other examples
This project is part of a bundle of three sideprojects focused on using transformers from HuggingFace in practice.

- [arXiv summarizer](https://github.com/VerleysenNiels/arxiv-summarizer) 
- [Image captioning](https://github.com/VerleysenNiels/image-captioning) (this repository)
- [Historical map segmentation]() (coming soon)
