import os
import json
import numpy as np
import time
import datetime
import argparse
import random
from tqdm import tqdm
from pathlib import Path

from prettytable import PrettyTable
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from transformers import BertTokenizer

import utils
from models.model_search import Search
from dataset import create_dataset, create_loader

# #################################################################################
# ### Evaluation Functions (from snippet 2)
# #################################################################################
@torch.no_grad()
def evaluation_itc(model, data_loader, tokenizer, device, config):
    """提取 ITC 特征 (Image-Text Contrastive)"""
    model.eval()
    print('Computing ITC features...')
    start_time = time.time()

    # 1. 提取文本特征
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config.get('batch_size_test_text', 256)
    
    text_embeds, text_atts, text_feats = [], [], []
    for i in tqdm(range(0, num_text, text_bs), desc="Text Features"):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(
            text, padding='max_length', truncation=True, 
            max_length=config['max_tokens'], return_tensors="pt"
        ).to(device)
        
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.get_text_feat(text_embed)
        text_feat = F.normalize(text_feat, dim=-1)

        text_embeds.append(text_embed.cpu())
        text_atts.append(text_input.attention_mask.cpu())
        text_feats.append(text_feat.cpu())

    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    # 2. 提取图像特征 + 构建图片路径
    image_embeds, image_feats = [], []
    img_paths = []  # 新增：收集完整图片路径
    
    for image, pose, img_idx in tqdm(data_loader, desc="Image Features"):
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)

        if config.get('be_pose_img', False):
            pose = pose.to(device)
            if model.be_pose_conv:
                pose = model.pose_conv(pose)
            pose_embed, _ = model.get_vision_embeds(pose)
            image_embed = model.pose_block(image_embed, pose_embed)

        image_feat = model.get_image_feat(image_embed)
        image_feat = F.normalize(image_feat, dim=-1)
        
        image_embeds.append(image_embed.cpu())
        image_feats.append(image_feat.cpu())
        
        # ✅ 关键修改：构建完整图片路径
        for idx in img_idx:
            idx_int = idx.item() if torch.is_tensor(idx) else idx
            img_relative_path = data_loader.dataset.ann[idx_int]['image']
            img_full_path = os.path.join(data_loader.dataset.image_root, img_relative_path)
            img_paths.append(img_full_path)

    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    # 3. 计算相似度矩阵 (ITC)
    sims_matrix = image_feats @ text_feats.t()
    sims_matrix_t2i = sims_matrix.t()

    total_time = time.time() - start_time
    print(f'✅ ITC features computed in {datetime.timedelta(seconds=int(total_time))}')

    return sims_matrix_t2i, image_embeds, text_embeds, text_atts, img_paths  # ✅ 返回 img_paths
@torch.no_grad()
def evaluation_itm(model, device, config, args, sims_matrix, image_embeds, text_embeds, text_atts):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing matching score')
    start_time = time.time()

    # Ensure all tensors participating in cross-modal scoring are on the same device
    image_embeds = image_embeds.to(device)
    text_embeds = text_embeds.to(device)
    text_atts = text_atts.to(device)
    sims_matrix = sims_matrix.to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0, device=device)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 500, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.get_cross_embeds(encoder_output, encoder_att,
                                        text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1),)[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    for i in range(sims_matrix.size(0)):
        score_matrix_t2i[i][score_matrix_t2i[i] == 1000.0] = replacement_tensor[i][score_matrix_t2i[i] == 1000.0]
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())

    score_sim_t2i = sims_matrix.clone()
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i  #

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Computing matching score time {}'.format(total_time_str))
    return score_matrix_t2i.cpu().numpy()


def mAP(scores_t2i, g_pids, q_pids, table=None):
    similarity = torch.tensor(scores_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :10].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP, _ = all_cmc, mAP, mINP, indices
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    if not table:
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        print(table)

    eval_result = {'R1': t2i_cmc[0],
                   'R5': t2i_cmc[4],
                   'R10': t2i_cmc[9],
                   'mAP': t2i_mAP,
                   'mINP': t2i_mINP,
                   }

    return eval_result

# #################################################################################
# ### Main Evaluation Logic
# #################################################################################

def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    print("### output_dir:", args.output_dir)
    
    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)

    # Load checkpoint for evaluation
    print(f"### Loading checkpoint from {args.checkpoint}")
    model.load_pretrained(args.checkpoint)
    
    model = model.to(device)
    print("Total Params: ", sum(p.numel() for p in model.parameters()))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("### Creating evaluation dataset")
    # We pass evaluate=True to create_dataset
    _, test_dataset = create_dataset(config, evaluate=True)

    start_time = time.time()

    print("### Start evaluating")
    test_loader = create_loader([test_dataset], [None],
                                batch_size=[config['batch_size_test']],
                                num_workers=[4],
                                is_trains=[False],
                                collate_fns=[None])[0]
    
    # --- Evaluation Pipeline ---
    sims_matrix_t2i, image_embeds, text_embeds, text_atts,img_paths = evaluation_itc(
        model_without_ddp, test_loader, tokenizer, device, config)
    
    score_test_t2i = evaluation_itm(model_without_ddp, device, config, args,
                                    sims_matrix_t2i, image_embeds, text_embeds, text_atts)
    
    if utils.is_main_process():
        print('### Evaluating result:')
        eval_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids)
        
        # Log results to a JSON file
        log_stats = {**{f'test_{k}': float(v) for k, v in eval_result.items()}}
        with open(os.path.join(args.output_dir, "evaluate.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        print(log_stats)
        
    if args.distributed:  # <--- 添加这个 if 判断
        dist.barrier()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Evaluation Complete. Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, help="Task identifier, used by config")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save logs")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false', help="Use distributed training")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'))
    
    # Save a copy of the config used for this evaluation
    yaml.dump(config, open(os.path.join(args.output_dir, 'config_eval.yaml'), 'w'))

    main(args, config)