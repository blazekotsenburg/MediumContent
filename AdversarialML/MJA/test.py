import os

import pandas as pd

from models.prompt import Prompt
from dotenv import load_dotenv
from dataclasses import dataclass

import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import random, hashlib, numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.decomposition import PCA
from scipy.stats import norm

from common.orchestrator import Orchestrator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Same backbone the paper used: ViT-L/14
TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
TEXT_MODEL = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")#.to(DEVICE)
TEXT_MODEL.eval()

@dataclass
class APOResult:
    sensitive_prompt: str
    best_prompt: str
    best_score: float
    queries: int

load_dotenv(dotenv_path="/Users/blazekotsenburg/Documents/Source/Repos/MediumContent/AdversarialML/MJA/.env")
PATH_METAPHOR_SYS_PROMPT=os.getenv("PATH_METAPHOR_SYS_PROMPT")
PATH_CONTEXT_SYS_PROMPT=os.getenv("PATH_CONTEXT_SYS_PROMPT")
PATH_ADV_SYS_PROMPT=os.getenv("PATH_ADV_SYS_PROMPT")

PATH_METAPHOR_USR_PROMPT=os.getenv("PATH_METAPHOR_USR_PROMPT")
PATH_CONTEXT_USR_PROMPT=os.getenv("PATH_CONTEXT_USR_PROMPT")
PATH_ADV_USR_PROMPT=os.getenv("PATH_ADV_USR_PROMPT")

# p = Prompt.load_from_file(file_path=PATH_METAPHOR_PROMPT)
# print(p.render())

sys_prompt_metaphor   = Prompt.load_from_file(file_path=PATH_METAPHOR_SYS_PROMPT)
sys_prompt_context    = Prompt.load_from_file(file_path=PATH_CONTEXT_SYS_PROMPT)
sys_prompt_adverarial = Prompt.load_from_file(file_path=PATH_ADV_SYS_PROMPT)

usr_prompt_metaphor     = Prompt.load_from_file(file_path=PATH_METAPHOR_USR_PROMPT)
usr_prompt_context      = Prompt.load_from_file(file_path=PATH_CONTEXT_USR_PROMPT)
usr_prompt_adversarial  = Prompt.load_from_file(file_path=PATH_ADV_USR_PROMPT)


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# llama_3_8b = AutoModelForCausalLM.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# llama_3_8b=None
# llama_3_8b = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )
llama_3_8b = Orchestrator(model_id="meta-llama/Meta-Llama-3-8B-Instruct")

# terminators = [
#     llama_3_8b.tokenizer.eos_token_id,
#     llama_3_8b.tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

def metaphors(x_sen: str, N: int) -> List[str]:
    metaphors = []
    for _ in range(N):

        messages = [
            {"role": "system", "content": sys_prompt_metaphor.render()},
            {"role": "user", "content": usr_prompt_metaphor.render(sen_content=x_sen)}
        ]
        metaphors.append(llama_3_8b.invoke(messages=messages))
    return metaphors

def contexts(x_sen: str, metaphor: str, M: int) -> List[str]:
    contexts = []
    for _ in range(M):
        messages = [
            {"role": "system", "content": sys_prompt_context.render()},
            {"role": "user", "content": usr_prompt_context.render(sen_content=x_sen, metaphor=metaphor)}
        ]
        contexts.append(llama_3_8b.invoke(messages=messages))
    return contexts

def adversarial(x_sen: str, metaphor:str, ctx: str) -> List[str]:
    messages = [
        {"role": "system", "content": sys_prompt_adverarial.render()},
        {"role": "user", "content": usr_prompt_adversarial.render(sen_content=x_sen, metaphor=metaphor, artistic_context=ctx)}
    ]

    return llama_3_8b.invoke(messages=messages)

def t2i_generate(prompt: str) -> str:
    """Call your T2I model here (stub returns prompt as 'image')."""
    return f"IMAGE({prompt})"

def safety_filter(image: str) -> bool:
    """Return True if NOT blocked (0 in indicator)."""
    return random.random() > 0.3  # 70 % chance to pass

def image_text_similarity(image: str, x_sen: str) -> float:
    """Return similarity score (CLIP, etc.)."""
    # hash-based pseudo‑similarity for demo only
    h_img = int(hashlib.sha256(image.encode()).hexdigest(), 16) % 1000
    h_txt = int(hashlib.sha256(x_sen.encode()).hexdigest(), 16) % 1000
    return 1 - abs(h_img - h_txt) / 1000

# ----------------------- LHS (loose implementation) --------------------- #
def lhs_split(items: List[str], n_obs: int) -> Tuple[List[str], List[str]]:
    """Simple Latin‑Hypercube‑like split: shuffle then take first n_obs."""
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled[:n_obs], shuffled[n_obs:]

# --------------- Ground‑truth evaluation for a prompt ------------------- #
def ground_truth(prompt: str, x_sen: str) -> float:
    """Return score = similarity * pass‑indicator (0 if blocked)."""
    img = t2i_generate(prompt)
    if not safety_filter(img):
        return 0.0
    return image_text_similarity(img, x_sen)

def embed(prompt: str, dim: int = 256) -> np.ndarray:
    """
    Convert a text prompt into a 768-dim CLIP embedding (NumPy, CPU).
    Works with Hugging Face 'openai/clip-vit-large-patch14'.
    """
    # 1. Tokenise; returns a dict of tensors
    tokens = TOKENIZER(
        prompt,
        truncation=True,
        padding="max_length",   # CLIP expects exactly 77 tokens
        max_length=77,
        return_tensors="pt"
    ).to(DEVICE)

    # 2. Forward pass through CLIP text encoder
    outputs = TEXT_MODEL(**tokens)

    # 3. Take the *pooled* text embedding (CLS token at position 0)
    text_emb = outputs.last_hidden_state[:, 0, :]   # shape [1, 768]

    # 4. L2-normalise so cosine-sim == dot product
    text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=-1)

    # 5. Move to CPU & flatten → NumPy row
    return text_emb.squeeze(0).cpu().numpy()

def run_apo(x_sen: str) -> APOResult:
    for met in metaphors(x_sen=x_sen, N=N):
        for ctx in contexts(x_sen=x_sen, metaphor=met, M=M):
            candidates.append(adversarial(x_sen=x_sen, metaphor=met, ctx=ctx))

    # 2) Initial observation / candidate split
    obs_prompts, can_prompts = lhs_split(candidates, min(N_OBS, len(candidates)))
    obs_scores = [ground_truth(p, x_sen) for p in obs_prompts]

    # Early success check
    best_idx = int(np.argmax(obs_scores))
    best_prompt, best_score = obs_prompts[best_idx], obs_scores[best_idx]
    if best_score >= SIM_THRESHOLD:
        return APOResult(x_sen, best_prompt, best_score, len(obs_prompts))

    no_improve = 0
    total_queries = len(obs_prompts)

    # --- Bayesian optimisation loop --- #
    while can_prompts:
        # Feature extraction + dimensionality reduction
        X_emb = np.array([embed(p) for p in obs_prompts])
        X_emb = PCA(n_components=min(50, X_emb.shape[1])).fit_transform(X_emb)

        pca = PCA(n_components=50).fit(X_emb)   #   <-- fit ONCE
        X_emb_reduced = pca.transform(X_emb)    #   <-- train GPR on this

        # Fit surrogate
        gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        gpr.fit(X_emb_reduced, obs_scores)

        # Predict μ, σ for candidates
        mu, sigma = [], []
        for p in can_prompts:
            # vec = PCA(n_components=min(50, X_emb.shape[1])).fit_transform(embed(p).reshape(1, -1))
            vec = embed(p).reshape(1, -1)       # CLIP → 768-dim
            vec = pca.transform(vec)  
            m, s = gpr.predict(vec, return_std=True)
            mu.append(m.item())
            sigma.append(s.item())

        mu, sigma = np.array(mu), np.array(sigma)
        Z = (mu - best_score) / (sigma + 1e-9)
        ei = (mu - best_score) * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Select best EI candidate
        best_can_idx = int(np.argmax(ei))
        next_prompt = can_prompts.pop(best_can_idx)

        # Real query
        next_score = ground_truth(next_prompt, x_sen)
        total_queries += 1

        # Update observation sets
        obs_prompts.append(next_prompt)
        obs_scores.append(next_score)

        # Check improvement / success
        if next_score > best_score:
            best_score, best_prompt = next_score, next_prompt
            no_improve = 0
        else:
            no_improve += 1

        if best_score >= SIM_THRESHOLD:
            break
        if no_improve >= EARLY_STOP_ROUNDS:
            break
        
    return APOResult(x_sen, best_prompt, best_score, total_queries)

df = pd.read_csv("/Users/blazekotsenburg/Documents/Source/Repos/MediumContent/AdversarialML/MJA/data/mja_dataset_2.csv")

rows={
    "idx": [],
    "sen_content": [],
    "metaphor":[],
    "context": [],
    "adversarial": []
}

N                 = 7
M                 = 6
N_OBS             = 8
EARLY_STOP_ROUNDS = 7    # R in the paper
SIM_THRESHOLD     = 0.85  # τ in the paper

candidates=[]
for idx, row in df.iterrows():
    x_sen = row["content"]
    run_apo(x_sen=x_sen)
    
    
# for idx, row in df.iterrows():
#     rows["idx"] = idx

#     sen_content = row["content"]
#     rows["sen_content"] = sen_content
#     messages = [
#         {"role": "system", "content": sys_prompt_metaphor.render()},
#         {"role": "user", "content": usr_prompt_metaphor.render(sen_content=sen_content)}
#     ]
#     print(messages)

#     metaphor_result = llama_3_8b.invoke(messages=messages)
#     rows["metaphor"].append(metaphor_result)

#     messages = [
#         {"role": "system", "content": sys_prompt_context.render()},
#         {"role": "user", "content": usr_prompt_context.render(sen_content=sen_content, metaphor=metaphor_result)}
#     ]
#     print(messages)

#     context_result = llama_3_8b.invoke(messages=messages)
#     rows["context"].append(context_result)

#     messages = [
#         {"role": "system", "content": sys_prompt_adverarial.render()},
#         {"role": "user", "content": usr_prompt_adversarial.render(sen_content=sen_content, metaphor=metaphor_result, artistic_context=context_result)}
#     ]

#     adversarial_result = llama_3_8b.invoke(messages=messages)
#     rows["adversaril"].append(adversarial_result)

# df_candidates = pd.DataFrame.from_dict(data=rows)
# df_candidates.to_csv("data_full.csv")