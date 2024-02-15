# Required Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial

# Initialize Model and Tokenizer
def initialize_model_and_tokenizer(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="cpu")
    use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    return model, tokenizer

# Utility Functions
def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

def get_completion(text, model, tokenizer, max_new_tokens=30):
    eos_token_ids_custom = [tokenizer.eos_token_id]
    with torch.no_grad():
        output = model.generate(
            **tokenizer(text, return_tensors='pt').to(model.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_ids_custom,
            do_sample=False
        )
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    return completion

# Patching Hook and Cache Handling
def patching_hook(activation, hook, cache, position, **kwargs):
    activation[:, position, :] = cache[hook.name][:, position, :]
    return activation

def interpolation_hook(activation, hook, cache, position, alpha=0.5, **kwargs):
    activation[:, position, :] = (1-alpha) * activation[:, position, :] + alpha * cache[hook.name][:, position, :]
    return activation

def clean_toxic_logit_diff(logits, clean_token_id, toxic_token_id):
    return logits[0, -1, clean_token_id] - logits[0, -1, toxic_token_id]  # Assuming 315 is 'CLEAN' and 7495 is 'TOXIC' token IDs


def get_resid_cache_from_forward_pass(model, tokens):
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    logits = logits.cpu()
    resid_cache = {}

    # Filter out resid caches
    for key in cache.keys():
        if key.endswith("hook_resid_post"):
            resid_cache[key] = cache[key].cpu()  # Ensure cache data is moved to CPU

    return logits, resid_cache


# Main function to process data and generate outputs
def run_patching_experiment_with_hook(model, tokens, resid_caches, clean_token_id, toxic_token_id, hook=None, **hook_kwargs):
    hook = patching_hook if hook is None else hook
    results = []
    for layer in tqdm(range(model.cfg.n_layers)):
        model.reset_hooks()
        temp_hook = partial(
            hook,
            cache=resid_caches,
            position=-1,  # Assuming we're interested in the last token's position
            **hook_kwargs
        )
        model.blocks[layer].hook_resid_post.add_hook(temp_hook)

        with torch.no_grad():
            logits = model(tokens).to("cpu")
            logit_diff_change = clean_toxic_logit_diff(logits, clean_token_id, toxic_token_id)
        results.append(logit_diff_change.item())

    return results

# Visualization
def plot_logit_differences(results):
    plt.title("Logit Differences (Clean - Toxic)")
    plt.xlabel("Layer")
    plt.ylabel("Logit Difference")
    plt.plot(results)
    plt.show()


if __name__ == "__main__":
    # Main Execution
    model_name_or_path = "meta-llama/Llama-2-13b-chat-hf"
    model, tokenizer = initialize_model_and_tokenizer(model_name_or_path)
    clear_gpu(model)  # Clear GPU cache if needed

    # Further processing and function calls go here, like processing data, applying hooks, and plotting results.
