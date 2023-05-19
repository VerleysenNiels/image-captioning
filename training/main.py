"""
Main script for finetuning a transformer model on a dataset. In this case I will be training a ViT-GPT-2 model to predict the prompt used to generate a given image with stable diffusion.
This script will be running on CUDA with PyTorch as deep learning framework.    
"""

import logging
logging.basicConfig(filename="logs/training.log" ,level=logging.INFO)

# Import pytorch in order to use gpu
import torch

# To download and manage the dataset from HuggingFace
from datasets import load_dataset

# Of course I'm using transformers to do the heavy lifting of the fintetuning process
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)

# Need to split the dataset in training and validation
from sklearn.model_selection import train_test_split


if __name__ == "__main__":    
    # Can we use gpu?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on {device}")
    
    logging.info("Loading dataset...")
    # Load the dataset
    prompt_dataset = load_dataset('poloclub/diffusiondb', 'large_random_10k', cache_dir="cache")
    
    logging.info("Loading model and tokenizer...")
    # We can automatically load in the tokenizer and model that we want to use
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model.to(device)
    
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    logging.info("Preparing dataset...")
    # We can then tokenize our dataset
    def tokenize_function(samples):
        inputs = feature_extractor(samples['image'], return_tensors='pt')
        prompts = tokenizer(samples['prompt'], truncation=True)
        
        inputs["labels"] = prompts["input_ids"]
        return inputs
    
    tokenized_prompt_dataset = prompt_dataset.map(tokenize_function, batched=True)
    
    logging.info("Splitting dataset into train and validation sets...")
    # Split the tokenized dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(tokenized_prompt_dataset, test_size=0.2, random_state=42)
    
    logging.info("Preparing data collator...")
    # Define the data collator with masked language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    logging.info("Preparing trainer...")
    # Next up let's define our training arguments    
    training_arguments = TrainingArguments(
        output_dir="vit-gpt2-image-to-prompt",  # Where to store results
        overwrite_output_dir=True,              # Overwrite the content of the output directory
        num_train_epochs=10,                    # Total number of training epochs
        per_device_train_batch_size=8,          # Batch size per GPU for training
        per_device_eval_batch_size=8,           # Batch size per GPU for evaluation
        save_strategy="epoch",                  # Saving strategy for checkpoints
        save_total_limit=3,                     # Maximum number of checkpoints to save
        evaluation_strategy="epoch",            # Evaluate the model every epoch
        logging_steps=100,                      # Number of training steps before logging training metrics
        learning_rate=1e-4,                     # Initial learning rate for the optimizer
        weight_decay=0.01,                      # Weight decay for the optimizer
        gradient_accumulation_steps=1,          # Number of steps before performing gradient accumulation
        fp16=True,                              # Whether to use mixed precision training with FP16
        dataloader_num_workers=4,               # Number of worker processes for data loading
        metric_for_best_model="loss",           # Metric to use for identifying the best model
        greater_is_better=False,                # Whether a higher value of the metric is better for the best model selection
        report_to='tensorboard',                # Always nice for following up on the training process
        push_to_hub=True                        # Push the results to the hub
    )
    
    # Finally we can set up the trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,        
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # And we can run our trainer
    logging.info("Started training...")
    result = trainer.train(resume_from_checkpoint = True)
    logging.info("Done")
    logging.info(f"Time: {result.metrics['train_runtime']:.2f}")
    logging.info(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

    logging.info(f"Pushing to hub...")
    trainer.push_to_hub()
    
    
    
    
