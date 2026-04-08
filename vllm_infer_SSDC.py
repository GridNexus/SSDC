import argparse
import base64
from io import BytesIO
import json
import os
import sys
import gc
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from prettytable import PrettyTable

# Ensure CMP modules are importable when running from ARR1/IRRA directory
CMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'CMP'))
if CMP_ROOT not in sys.path:
    sys.path.insert(0, CMP_ROOT)

from vllm import LLM, SamplingParams
from transformers import AutoProcessor, BertTokenizer
from qwen_vl_utils import process_vision_info

from datasets import build_dataloader 
from datasets.bases import TextPureDataset  
from utils.logger import setup_logger
from utils.iotools import load_train_configs
from model import build_model, build_clip_model
from utils.checkpoint import Checkpointer
import os.path as op

# ===== CMP embedding dependencies (added for CMP adaptation) =====
from ruamel.yaml import YAML
from models.model_search import Search
from dataset import create_dataset, create_loader

yaml = YAML(typ="safe")

# ==================== 保持原有的辅助函数 ====================
def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[: , : max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    
    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[: , i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices

def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    if retur_indices:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]], indices
    else:
        return [n_, t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP, t2i_cmc[0]+ t2i_cmc[4]+ t2i_cmc[9]]

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def process_cap_(caps):
    tmps = []
    for c in caps:
        c = c.split('\n') 
        tmp = []
        for cc in c:
            if ': ' in cc:
                continue
            try:
                cc = cc.split('.')[1]
                if 'Yes, ' in cc:
                    cc = cc.replace('Yes, ','')
                if 'No, ' in cc:
                    cc = cc.replace('No, ','')
                    
                cc = cc[: 1].upper() + cc[1:]
                if cc[-1:] != '.':
                    cc += '.'
            except:
                cc = '' 
            tmp.append(cc) 
        tmps.append(tmp) 
    return tmps

def print_rs(sims_dict, qids, pids, logger):
    table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP","rSum"])
    for key in sims_dict.keys():
        sims = sims_dict[key]
        rs = get_metrics(sims, qids, pids, f'{key}-t2i', False)
        table.add_row(rs) 

    table.custom_format["R1"] = lambda f, v: f"{v:.2f}"
    table.custom_format["R5"] = lambda f, v: f"{v:.2f}"
    table.custom_format["R10"] = lambda f, v: f"{v:.2f}"
    table.custom_format["mAP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["mINP"] = lambda f, v: f"{v:.2f}"
    table.custom_format["rSum"] = lambda f, v: f"{v:.2f}"
    logger.info('\n' + str(table))

# ==================== 单卡 vLLM 类（核心修改）====================
class MLLMs(object):
    def __init__(self, model_dir):
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"
        
        # 🔥 强制单卡
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        self.model_dir = model_dir  
        print("="*60)
        print("🚀 Loading vLLM Model (Single GPU Mode)...")
        print(f"📁 Model Path: {model_dir}")
        print("="*60)
        
        self.llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,  # 🔥 强制单GPU
            gpu_memory_utilization=0.7,  # 保持原设置
            max_num_seqs=256,
            max_model_len=1536,
            enforce_eager=True,
            disable_log_stats=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            dtype=torch.bfloat16
        )
        
        self.processor = AutoProcessor.from_pretrained(model_dir)
        print("✅ vLLM Model Loaded Successfully (Single GPU)")
        print("="*60)

    def generate_response_multi_images(self, questions, images=None, sys="You are a helpful assistant.", t=0.01):
        try:
            messages = [ 
                [{"role": "system", "content": sys}, {"role": "user",  "content": [
                    {"type": "image", "image": images[i], "min_pixels": 50176, "max_pixels": 50176},
                    {"type":  "text", "text": p}
                ]}] 
                for i, p in enumerate(questions)]
            
            prompts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]   
            image_data = [process_vision_info(msg)[0] for msg in messages]
        
            inputs = [{
                "prompt": p,
                "multi_modal_data": {"image": image_data[i]} 
            } for i, p in enumerate(prompts)]
            
            sampling_params = SamplingParams(temperature=t, max_tokens=512, skip_special_tokens=True)
            outputs = self.llm.generate(inputs, sampling_params=sampling_params)
            results = [o.outputs[0].text for o in outputs]
            
            # 清理
            del inputs, outputs, image_data, messages, prompts
            torch.cuda.empty_cache()
            gc.collect()
            
            return results
            
        except Exception as e: 
            print(f"❌ 生成失败:  {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [""] * len(questions)
    
    def generate_response_qwen2vl(self, questions, sys="You are a helpful assistant.", t=0.01):
        try:
            messages = [ 
                [{"role": "system", "content": sys}, {"role":  "user",  "content": [{"type": "text", "text":  p}]}] 
                for p in questions]
            
            prompts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages] 
            inputs = [{"prompt": p} for p in prompts]
            
            sampling_params = SamplingParams(temperature=t, max_tokens=512, skip_special_tokens=True)
            outputs = self.llm.generate(inputs, sampling_params=sampling_params)
            results = [o.outputs[0].text for o in outputs]
            
            del inputs, outputs, messages, prompts
            torch.cuda.empty_cache()
            gc.collect()
            
            return results
            
        except Exception as e:
            print(f"❌ 文本生成失败: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            return [""] * len(questions)

# ==================== Dataset 类 ====================
class ImgDataset(Dataset):
    def __init__(self, images):  
        self.images = images
    def __len__(self):
        return len(self.images) 
    def __getitem__(self, index):
        return {'index': index, 'images': load_image(self.images[index])}

class TxtDataset(Dataset):
    def __init__(self, captions):  
        self.captions = captions
    def __len__(self):
        return len(self.captions) 
    def __getitem__(self, index):
        return {'index': index, 'captions': self.captions[index]}

def collate(batch):
    keys = set([key for b in batch for key in b.keys()]) 
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            batch_tensor_dict.update({k: list(v)})  
    return batch_tensor_dict

# ==================== 批量推理函数 ====================
def batch_infer(llm, b_prompts, images, t=0.2):
    results = []
    n_samples = len(b_prompts)
    micro_batch = 8
    n_batches = (n_samples - 1) // micro_batch + 1
    
    print(f"📊 总样本数: {n_samples}, 分成 {n_batches} 个批次")
    
    for i in tqdm(range(n_batches), desc="Batch Inference"): 
        start = i * micro_batch
        end = min(start + micro_batch, n_samples)
        
        try:
            rs = llm.generate_response_multi_images(
                questions=b_prompts[start: end], 
                images=images[start:end], 
                t=t
            )
            
            if rs and len(rs) > 0:
                print(f"✓ Batch {i+1}/{n_batches}:  {rs[0][: 80]}...")
            results += rs
            
            torch.cuda.empty_cache()
            gc.collect()
            
            if (i + 1) % 5 == 0:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"❌ Batch {i+1} 失败: {e}")
            results += [""] * (end - start)
            torch.cuda.empty_cache()
            gc.collect()
    
    return results

def batch_infer_txt(llm, b_prompts, t=0.2):
    results = []
    n_samples = len(b_prompts)
    micro_batch = 16
    n_batches = (n_samples - 1) // micro_batch + 1
    
    print(f"📊 文本样本数: {n_samples}, 分成 {n_batches} 个批次")
    
    for i in tqdm(range(n_batches), desc="Text Inference"): 
        start = i * micro_batch
        end = min(start + micro_batch, n_samples)
        
        try:
            rs = llm.generate_response_qwen2vl(questions=b_prompts[start: end], t=t)
            
            if rs and len(rs) > 0:
                print(f"✓ Text Batch {i+1}/{n_batches}: {rs[0][:80]}...")
            results += rs
            
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"❌ Text Batch {i+1} 失败: {e}")
            results += [""] * (end - start)
    
    return results

# ==================== CMP Embedding Helpers ====================
def load_cmp_components(config_path, checkpoint_path, device):
    """Load CMP tokenizer/model exactly as vllm_infer_cmpv5 does."""
    cmp_config = yaml.load(open(config_path, 'r'))
    
    def _resolve(path):
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(CMP_ROOT, path)

    if 'vision_config' in cmp_config: 
        cmp_config['vision_config'] = _resolve(cmp_config['vision_config'])
    if 'text_config' in cmp_config:
        cmp_config['text_config'] = _resolve(cmp_config['text_config'])

    tokenizer = BertTokenizer.from_pretrained(cmp_config['text_encoder'])
    model = Search(config=cmp_config)
    model.load_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return cmp_config, tokenizer, model

@torch.no_grad()
def extract_cmp_features(model, data_loader, tokenizer, device, config):
    """Adapted from CMP pipeline:  generate normalized text/image embeddings."""
    model.eval()
    texts = data_loader.dataset.text
    text_bs = config.get('batch_size_test_text', 256)
    text_feats_list, text_embeds_list, text_atts_list = [], [], []

    for i in tqdm(range(0, len(texts), text_bs), desc="Text Features"):
        batch_text = texts[i:  i + text_bs]
        text_input = tokenizer(
            batch_text,
            padding='max_length',
            truncation=True,
            max_length=config['max_tokens'],
            return_tensors="pt",
        ).to(device)

        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.get_text_feat(text_embed)

        text_embeds_list.append(text_embed.cpu())
        text_atts_list.append(text_input.attention_mask.cpu())
        text_feats_list.append(F.normalize(text_feat, dim=-1).cpu())

    image_feats_list, image_embeds_list, img_paths = [], [], []
    for image, pose, img_idx in tqdm(data_loader, desc="Image Features"):
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)

        if config.get('be_pose_img', False) and pose is not None:
            pose = pose.to(device)
            pose_embed, _ = model.get_vision_embeds(model.pose_conv(pose))
            image_embed = model.pose_block(image_embed, pose_embed)

        image_feat = model.get_image_feat(image_embed)
        image_embeds_list.append(image_embed.cpu())
        image_feats_list.append(F.normalize(image_feat, dim=-1).cpu())

        for idx in img_idx:
            path = data_loader.dataset.ann[idx.item()]['image']
            img_paths.append(os.path.join(data_loader.dataset.image_root, path))

    return {
        'sims_itc': torch.cat(text_feats_list, dim=0) @ torch.cat(image_feats_list, dim=0).t(),
        'text_feats': torch.cat(text_feats_list, dim=0),
        'image_feats': torch.cat(image_feats_list, dim=0),
        'text_embeds':  torch.cat(text_embeds_list, dim=0),
        'image_embeds': torch.cat(image_embeds_list, dim=0),
        'text_atts': torch.cat(text_atts_list, dim=0),
        'img_paths': img_paths,
        'texts': texts,
    }

@torch.no_grad()
def compute_cmp_itm_scores(model, features, device, config):
    """Standalone ITM logic borrowed from CMP实现 (single process version)."""
    sims_matrix = features['sims_itc'].to(device)
    image_embeds = features['image_embeds'].to(device)
    text_embeds = features['text_embeds'].to(device)
    text_atts = features['text_atts'].to(device)

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0, device=device)
    k_test = config.get('k_test', 128)

    for i, sims in enumerate(tqdm(sims_matrix, desc="ITM Re-ranking")):
        topk_sim, topk_idx = sims.topk(k=min(k_test, sims.size(0)), dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long, device=device)

        current_text_embed = text_embeds[i].unsqueeze(0).repeat(encoder_output.size(0), 1, 1)
        current_text_att = text_atts[i].unsqueeze(0).repeat(encoder_output.size(0), 1)

        output = model.get_cross_embeds(
            encoder_output,
            encoder_att,
            text_embeds=current_text_embed,
            text_atts=current_text_att,
        )[:, 0, :]

        score = model.itm_head(output)[: , 1]
        score_matrix_t2i[i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    mask = score_matrix_t2i == 1000.0
    score_matrix_t2i[mask] = replacement_tensor[mask]

    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())
    score_sim_t2i = (sims_matrix - sims_matrix.min()) / (sims_matrix.max() - sims_matrix.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i

    return score_matrix_t2i.cpu()

def round_llm(args, round, llm, logger, img_paths, captions, rrcaptions, gt_indexs, sims_base, decisions, xi, n_round, pids, batch_size, rcaptions, ggt_indexs):
    k = n_round
    
    results = []
    n_samples = sims_base.size(0)
    n_batches = (n_samples - 1)//batch_size + 1
    agrmaxs = []
    for i in tqdm(range(n_batches)): 
        start = i*batch_size
        end = n_samples if i== n_batches -1 else (i+1)*batch_size   
        agrmaxs += sims_base[start:end].topk(dim=1, k=k)[1].numpy().tolist() 
    
    ref_images = [[img_paths[j] for j in i] for i in agrmaxs] 
    ref_indexs = [[j for j in i] for i in agrmaxs] 
    ref_pids = [[pids[j]  for j in i] for i in agrmaxs]
 
    prompt1 = """You are an expert in Person Re-identification and Anomaly Detection.
Task: Determine if the text accurately describes the **primary person** in the image.

Evaluation Criteria:
1.**Appearance**:  Check gender, clothing color, clothing type (upper/lower), and distinct accessories.
2.**Action**: Check if the described action (e.g., walking, falling, fighting) matches the person's behavior.
3.**Ignore**: Do not judge based on background details or lighting differences unless they obscure the person.

Text: {cap}

Does this text accurately describe the image?  Answer STRICTLY with "Yes" or "No"."""

    prompt2 = """According to the pedestrian image, answer the following questions one by one: 

1.The person is male or female?
2.What hairstyle does the person have, such as hair length and color?
3.What is this person wearing on his upper body?  If clearly visible, what are the color, type, and sleeve length? 
4.What are the characteristics of this person's pants?  If clearly visible, what are the color, type, and trouser leg length?
5.Does this person have any patterns on his/her clothes or pants?
6.What are the characteristics of this person's shoes?  If clearly visible, what are the color and style?
7.Does this person wear glasses?  If clearly visible, what are the color and style?
8.Is this person wearing a scarf?  If clearly visible, what are the color and style?
9.Does this person have something in his/her hand?  If so, what is it and what color is it?
10.Does this person carry a backpack? If clearly visible, what are the color and style?
11.Does this person wear a hat? If clearly visible, what are the color and style?
12.Is this person wearing a belt or waistband? 
13.What is this person doing? 
14.What is the background? 
15.Are there other people in the background of this person?"""
 
    prompt3 = """
    Task: Aggregate the following subtexts into a single continuous and concise text paragraph.
    Format:  Return the result strictly as a JSON object with a single key "caption".

    Requirements: 

    1.Grammar Flow: Ensure the transition after the prefix is natural.
    2.Keep the output concise, fluent and grammatical.
    3.The final returned content must be a JSON object with the single key "caption" whose value is the aggregated caption string.

    Now let's get started.
    Subtexts:  {cap}
    Output: 
    """

    top_indexs = [arg[0] for arg in agrmaxs]
    sims = [sims_base[i][ref_indexs[i][0]] for i, v in enumerate(gt_indexs)]
    conditions = [1 if rrcaptions[i]==captions[i] else 0 for i, v in enumerate(gt_indexs)]
    
    # Step 1
    images_stage1, prompts_stage1  = [], []
    for i, v in enumerate(gt_indexs):
        if conditions[i] == 1:
            images_stage1.append(ref_images[i][round])
            prompts_stage1.append(prompt1.format(cap=captions[i]))
    
    rss = batch_infer(llm, prompts_stage1, images_stage1, t=0.01)  
    rpl_ids  = [i for i, v in enumerate(gt_indexs) if conditions[i] == 1]

    for j, ids in enumerate(rpl_ids):
        if 'yes' in rss[j].lower():
            decisions[round][ids] = 1 
        else:
            decisions[round][ids] = 0

    rrpl_ids = []
    for j, ids in enumerate(rpl_ids):
        flg = 0
        for l in range(round):
            if decisions[l][ids] == 1:
                flg += 1  
        if decisions[round][ids] == 1 and round==0 and sims[ids] > xi:  
            gt_indexs[ids] = ref_indexs[ids][round]
            rrpl_ids.append(ids)

        if flg==0 and decisions[round][ids] == 1 and round>0 and sims[ids] <= xi:
            gt_indexs[ids] = ref_indexs[ids][round]
            ggt_indexs[ids] = ref_indexs[ids][round]
            rrpl_ids.append(ids)
    
    # Step 2    
    prompts_stage2 = [prompt2.format(cap=captions[v]) for v in rrpl_ids] 
    images_stage2  = [img_paths[gt_indexs[v]] for v in rrpl_ids]
    rs = batch_infer(llm, prompts_stage2, images_stage2, t=0.01)  
  
    rs = process_cap_(rs)
    for i, v in enumerate(rs): 
        rcaptions[rrpl_ids[i]] = v 

    prompts_stage3 = []
    for v in rrpl_ids: 
        subtexts = [captions[v]] + rcaptions[v]
        cap_json = json.dumps(subtexts, ensure_ascii=False)
        prompts_stage3.append(prompt3.format(cap=cap_json))

    rs = batch_infer_txt(llm, prompts_stage3, t=0.01)
    for i, v in enumerate(rs): 
        rrcaptions[rrpl_ids[i]] = v

# ==================== 关键修改：使用预先提取的特征进行评估 ====================
def eval_with_cached_features(args, qids, pids, round, t, logger, rrcaptions, gt_indexs, sims_base, global_sims, 
                              qfeats_cached, gfeats_cached, vq_feats_cached, vg_feats_cached, base_model):
    """使用缓存的 CLIP 特征进行评估，不需要重新加载模型"""
    
    sims_ = sims_base.clone()
    
    for i, g in enumerate(gt_indexs):
        if g > -1: 
            tmp = sims_[i].clone()
            tmp[g] = 1.0
            sims_[i] = tmp
        else:
            sims_[i] = sims_base[i].clone()

    sims_ = sims_base * t + (1 - t) * sims_
    
    sims_dict = {
        'sims_base': sims_base,
        'sims_last': global_sims,
        'sims_now': sims_,
    }
    
    print_rs(sims_dict, qids, pids, logger)
    return sims_

# ==================== 主函数 ====================

# ==================== 主函数 ====================
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="ICL Args")
    parser.add_argument("--base_model", default='RDE', type=str)
    parser.add_argument("--root_dir", default='', type=str)
    parser.add_argument("--embed_model_path", default='', type=str)
    parser.add_argument("--source", default='PAB', type=str)
    parser.add_argument("--target", default='PAB', type=str)
    parser.add_argument("--model_dir", default='', type=str)    
    parser.add_argument("--tag", default='', type=str)
    parser.add_argument("--lambda", default=0.8, type=float)
    parser.add_argument("--xi", default=0.5, type=float)
    parser.add_argument("--round", default=5, type=int)
    parser.add_argument("--ssdc_config", "--cmp_config", dest="ssdc_config", type=str, required=True,
                        help="YAML config for SSDC stage-1 model")
    parser.add_argument("--ssdc_checkpoint", "--cmp_checkpoint", dest="ssdc_checkpoint", type=str, required=True,
                        help="Checkpoint for SSDC stage-1 model")
    
    paras = parser.parse_args()
    paras = vars(paras)
    
    cmp_config_path = paras['ssdc_config']
    cmp_checkpoint_path = paras['ssdc_checkpoint']
    cmp_config = yaml.load(open(cmp_config_path, 'r'))
    
    base_model = paras['base_model']
    tttt = paras['lambda']
    xi = paras['xi']
    tag = f"{paras['source']}_{paras['target']}_xi{xi}_lam{tttt}_{base_model}{paras['tag']}"

    parser.add_argument("--config_file", default=f"{paras['embed_model_path']}/configs.yaml")
    args = parser.parse_args()
    args = load_train_configs(args. config_file)
    args. training = False
    args.root_dir = paras['root_dir']
    args. dataset_name = paras['target']
    args.model_dir = paras['model_dir']

    base = f"{paras['embed_model_path']}/{tag}_{args.dataset_name}/"
    os.makedirs(base, exist_ok=True)

    logger = setup_logger(base_model, save_dir=base, if_train=args.training) 
    logger.info(args)
    
    with open(f'{base}config.json', 'w', encoding='utf-8') as f:
        json.dump(paras, f, ensure_ascii=False, indent=4)
    
    # ==================== 🔥 阶段1：先提取 CMP 特征（vLLM 之前）====================
    logger.info("="*60)
    logger.info("阶段1：提取 CMP 特征 (单卡模式) - 在 vLLM 加载之前")
    logger.info("="*60)
    
    feature_cache = f'{base}/cmp_features.pt'
    device = torch.device("cuda:0" if torch.cuda. is_available() else "cpu")

    if not os.path.exists(feature_cache):
        logger.info("使用 CMP Search 模型提取特征")
        cmp_config_loaded, tokenizer, cmp_model = load_cmp_components(
            cmp_config_path, cmp_checkpoint_path, device
        )
        cmp_config = cmp_config_loaded
        _, cmp_dataset = create_dataset(cmp_config, evaluate=True)
        cmp_loader = create_loader(
            [cmp_dataset],
            [None],
            batch_size=[cmp_config['batch_size_test']],
            num_workers=[4],
            is_trains=[False],
            collate_fns=[None],
        )[0]

        feats = extract_cmp_features(cmp_model, cmp_loader, tokenizer, device, cmp_config)
        sims_base = compute_cmp_itm_scores(cmp_model, feats, device, cmp_config)

        qfeats = feats['text_feats']
        gfeats = feats['image_feats']
        vq_feats = None
        vg_feats = None
        img_paths = feats['img_paths']
        captions = feats['texts']
        qids = torch.tensor(cmp_dataset.q_pids)
        pids = torch. tensor(cmp_dataset.g_pids)

        torch.save(
            {
                'qfeats': qfeats. cpu(),
                'gfeats':  gfeats.cpu(),
                'vq_feats': None,
                'vg_feats': None,
                'sims_base': sims_base.cpu(),
                'img_paths': img_paths,
                'captions': captions,
                'q_pids': qids,
                'g_pids': pids,
            },
            feature_cache,
        )

        logger.info(f"✅ CMP 特征已保存到:  {feature_cache}")

        # 🔥 关键：完全释放 CMP 模型和显存
        del cmp_model, tokenizer, cmp_loader, cmp_dataset, feats
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ CMP 模型已释放，显存已清空")
        time.sleep(5)  # 等待显存完全释放
    else:
        logger.info(f"✅ 加载缓存特征: {feature_cache}")
        cached = torch.load(feature_cache)
        qfeats = cached['qfeats']
        gfeats = cached['gfeats']
        vq_feats = cached. get('vq_feats')
        vg_feats = cached.get('vg_feats')
        sims_base = cached['sims_base']
        img_paths = cached['img_paths']
        captions = cached['captions']
        if 'q_pids' in cached and 'g_pids' in cached:
            qids = cached['q_pids']
            pids = cached['g_pids']
            logger.info("✅ 缓存中包含 q_pids / g_pids，直接复用")
        else:
            logger.info("⚠️ 缓存缺少 q_pids / g_pids，重新创建数据加载器以获取")
            _, cmp_dataset = create_dataset(cmp_config, evaluate=True)
            qids = torch.tensor(cmp_dataset. q_pids)
            pids = torch.tensor(cmp_dataset.g_pids)

    # 初始化变量
    rcaptions = [[] for _ in captions]
    rrcaptions = copy.deepcopy(captions)
    gt_indexs = [-1 for _ in captions]
    ggt_indexs = [-1 for _ in captions]
    qids = torch.tensor(qids)
    pids = torch.tensor(pids)
    global_sims = sims_base.clone()

    # 打印初始 CMP 基线指标
    base_rs = get_metrics(global_sims.cpu(), qids, pids, 'CMP-Base', False)
    logger.info("初始 CMP 基线指标 (无 LLM): R1={:.2f} R5={:.2f} R10={:.2f} mAP={:.2f} mINP={:.2f} rSum={:.2f}".format(*base_rs[1:]))
    
    batch_size = 128
    
    # ==================== 🔥 阶段2：现在才加载 vLLM（CMP 已释放）====================
    logger. info("="*60)
    logger.info("阶段2：加载 vLLM 模型 (单GPU模式)")
    logger.info("="*60)
    
    # 确保显存完全释放后再加载 vLLM
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    
    llm = MLLMs(model_dir=args.model_dir)
    
    logger.info("="*60)
    logger.info("阶段3：开始 In-Context Learning 迭代")
    logger.info("="*60)
    
    n_round = paras['round']
    decisions = np.array([[0 for _ in captions] for rrr in range(n_round)])
    
    start_time = time.time()
    
    for round in range(n_round):
        logger.info("="*20 + f" Round-{round+1} " + "="*20)
        start_round_time = time.time()
        
        round_llm(args, round, llm, logger, img_paths, captions, rrcaptions, gt_indexs, sims_base, decisions, xi, n_round, pids, batch_size, rcaptions, ggt_indexs)
        
        global_sims = eval_with_cached_features(args, qids, pids, round, tttt, logger, rrcaptions, gt_indexs, sims_base, global_sims,
                                               qfeats, gfeats, vq_feats, vg_feats, base_model)
        
        end_round_time = time.time()
        round_time = end_round_time - start_round_time
        total_time = end_round_time - start_time
        
        logger.info(f"✅ Round {round+1} 完成")
        logger.info(f"  ⏱️  本轮耗时: {round_time:.0f}秒 ({round_time/60:.1f}分钟)")
        logger.info(f"  ⏱️  累计耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")
    
    total_time = time.time() - start_time
    logger.info("="*60)
    logger.info(f"🎉 所有轮次完成！总耗时: {total_time:.0f}秒 ({total_time/60:.1f}分钟)")
    logger.info("="*60)