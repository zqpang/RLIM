from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import shutil
import os
from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

import random

class CAVIAR(BaseImageDataset):

    dataset_dir = ''

    def __init__(self, root, verbose=True, **kwargs):
        super(CAVIAR, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        #self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_all')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        #gallery = self._process_gallery(self.gallery_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> CAVIAR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        #self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        #self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        #self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        
        #获取dir_path下所有jpg的路径
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 50  # pid == 0 means background
            assert 1 <= camid <= 2
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, int(pid), camid))

        return dataset


    def _process_gallery(self, dir_path, relabel=False):
        
        '''#获取dir_path下所有jpg的路径
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 50  # pid == 0 means background
            assert 1 <= camid <= 2
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid, 1))'''
            
        pattern = re.compile(r'([-\d]+)_c(\d)')
        img_paths = os.listdir(dir_path)
        dataset = []
        for pid in img_paths:
            paths_small = osp.join(dir_path,pid)
            images_paths_url = os.listdir(paths_small)
            index = random.randint(0,len(images_paths_url)-1)
            
            _, camid = map(int, pattern.search(images_paths_url[index]).groups())
            
            dataset.append((osp.join(paths_small,images_paths_url[index]),int(pid),camid))

        return dataset
