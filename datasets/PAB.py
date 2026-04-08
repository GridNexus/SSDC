import os.path as op
from typing import List
import logging
from glob import glob
import json  # 需要导入 json 库

from .bases import BaseDataset

class PAB(BaseDataset):
    """
    PAB Dataset
    """
    dataset_dir = 'PAB'

    def __init__(self, root='', verbose=True):
        super(PAB, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = self.dataset_dir 
        
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno()

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> PAB Images and Captions are loaded")
            self.show_dataset_info()

    def _split_anno(self):
        train_annos = []
        test_annos = []
        val_annos = []

        # --- 1. 加载训练集 (attr_0.json 到 attr_74.json) ---
        train_dir = op.join(self.dataset_dir, 'annotation', 'train')
        self.logger.info(f"Loading training annotations from: {train_dir}")
        
        json_files = sorted(glob(op.join(train_dir, 'attr_*.json')))
        
        if len(json_files) == 0:
             raise RuntimeError(f"No training json files found in {train_dir}")

        for fpath in json_files:
            # --- 核心修改：兼容 JSON Lines 格式 ---
            try:
                # 尝试作为普通 JSON 读取 (如果是标准列表格式)
                with open(fpath, 'r') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        train_annos.extend(content)
                    else:
                        train_annos.append(content)
            except json.JSONDecodeError:
                # 如果报错 "Extra data"，说明是 JSON Lines 格式
                # 我们逐行读取
                with open(fpath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                train_annos.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass # 跳过空行或损坏的行
            # ----------------------------------------

        # --- 2. 加载测试集 ---
        test_path = op.join(self.dataset_dir, 'annotation', 'test', 'attr.json')
        if op.exists(test_path):
            try:
                with open(test_path, 'r') as f:
                    test_annos = json.load(f)
            except json.JSONDecodeError:
                # 测试集也可能是 JSON Lines，做同样的处理
                with open(test_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            test_annos.append(json.loads(line))
        else:
            self.logger.warning(f"Test file not found: {test_path}")

        # 3. 验证集
        val_annos = test_annos

        return train_annos, test_annos, val_annos

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                # 容错处理：防止读取到空对象
                if not anno: continue

                pid = int(anno.get('image_id', anno.get('id')))
                pid_container.add(pid)
                
                img_path = op.join(self.img_dir, anno.get('image', anno.get('file_path')))
                
                captions = anno.get('caption', anno.get('captions'))
                
                if isinstance(captions, str):
                    captions = [captions]
                
                # 容错处理：如果caption是空的，跳过
                if not captions: continue

                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                
                image_id += 1
            
            # ReID 要求 PID 从 0 开始连续编号
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
            
            final_dataset = []
            for pid, img_id, path, cap in dataset:
                final_dataset.append((pid2label[pid], img_id, path, cap))
                
            return final_dataset, pid2label
            
        else:
            # 测试集处理
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            
            for anno in annos:
                if not anno: continue

                pid = int(anno.get('image_id', anno.get('id')))
                pid_container.add(pid)
                
                img_path = op.join(self.img_dir, anno.get('image', anno.get('file_path')))
                
                caption_list = anno.get('caption', anno.get('captions'))
                if isinstance(caption_list, str):
                    caption_list = [caption_list]
                
                if not caption_list: continue

                img_paths.append(img_path)
                image_pids.append(pid)

                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
                    
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))