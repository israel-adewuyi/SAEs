import torch
import os

from huggingface_hub import HfApi, HfFolder
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner, SAE, upload_saes_to_huggingface

def main():
    if torch.cuda.is_available():
        device = "cuda:3"
    else:
        device = "cpu"
    
    print("Using device:", device)

    total_training_steps = 200_000
    batch_size = 1024
    total_training_tokens = total_training_steps * batch_size
    
    lr_warm_up_steps = 0
    lr_decay_steps = total_training_steps // 5  # 20% of training
    l1_warm_up_steps = total_training_steps // 20  # 5% of training

    # layer = 23
    # width = 37
    
    for i in range(6):
        cfg = LanguageModelSAERunnerConfig(
            # Data Generating Function (Model + Training Distibuion)
            model_name="gpt2-medium", 
            hook_name=f"blocks.{i}.hook_resid_post", 
            hook_layer=i,  
            d_in=1024,  
            dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",  
            # dataset_path="NeelNanda/c4-code-20k",
            is_dataset_tokenized=True,
            streaming=True, 
            # SAE Parameters
            architecture="gated",
            mse_loss_normalization=None,  
            expansion_factor=32,  
            b_dec_init_method="geometric_median", 
            apply_b_dec_to_input=False,
            normalize_sae_decoder=False,
            scale_sparsity_penalty_by_decoder_norm=True,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=True,
            normalize_activations="expected_average_only_in",
            # Training Parameters
            lr=5e-5,  
            adam_beta1=0.9, 
            adam_beta2=0.999,
            lr_scheduler_name="constant", 
            lr_warm_up_steps=lr_warm_up_steps,  
            lr_decay_steps=lr_decay_steps,  
            l1_coefficient=5,  
            l1_warm_up_steps=l1_warm_up_steps, 
            lp_norm=1.0,  
            train_batch_size_tokens=batch_size,
            context_size=1024,
            # Activation Store Parameters
            n_batches_in_buffer=64, 
            training_tokens=total_training_tokens,  
            store_batch_size_prompts=16,
            # Resampling protocol
            use_ghost_grads=False, 
            feature_sampling_window=1000, 
            dead_feature_window=1000,  
            dead_feature_threshold=1e-4,  
            # WANDB
            log_to_wandb=True,
            wandb_project="GPT2_M_SAEs",
            wandb_log_frequency=20,
            eval_every_n_wandb_logs=10,
            # Misc
            device=device,
            seed=42,
            n_checkpoints=0,
            checkpoint_path="checkpoints",
            dtype="float32",
        )
    
        gated_sae = SAETrainingRunner(cfg).run()
    
        print("SAE Training completed.")
        print("------------- Uploading to HF now  -------------")
        
        saes_dict = {
            f"/blocks.{i}.hook_resid_post": gated_sae,
        }
    
        HfFolder.save_token("HF_TOKEN")
        
        upload_saes_to_huggingface(
            saes_dict,
            hf_repo_id="israel-adewuyi/GPT2_M_"
        )


if __name__ == "__main__":
    main()