import os
import pickle
import torch
from PIL import Image
import numpy as np
import json
import random
from preprocess.descriptors import SentenceEmbedModel

class VQADataset(torch.utils.data.Dataset):

    def __init__(self, root="data", transforms = None):
        self.root = root
        self.transforms = transforms
        self._load()
        self.text_embed_model = SentenceEmbedModel()

    def __getitem__(self, idx):
        descriptors = self.combos[idx]
        key = descriptors["Image"]
        slogan = self.text_embed_model.get_vector_rep(descriptors["Slogan"])
        slogan_id = self.text_embed_model.get_vector_rep(descriptors["Slogan id"])
        sentiment = self.text_embed_model.get_vector_rep(descriptors["Sentiment"])
        strategy = self.text_embed_model.get_vector_rep(descriptors["Strategy"])
        topic = self.text_embed_model.get_vector_rep(descriptors["Topic"])
        qa =self.text_embed_model.get_vector_rep(descriptors["QA"])

        filename = os.path.join(self.root, "{}".format(key))
        image = Image.open(filename, mode='r').convert('RGB')
        image = image.resize((501, 501))

        answer = descriptors["Slogan id"]

        # create target
        target = {}
        target["qa"] = qa
        target["sentiment"] = sentiment
        target["strategy"] = strategy
        target["slogan"] = slogan
        target["topic"] = topic
        target["image id"] = key
        # transforms image and target
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        answer = torch.Tensor([slogan_id])
        
        return image, target, answer
    
    
    def __len__(self) -> int:
        """Return the size of the dataset

        Returns:
            int: number of images in the dataset
        """
        return len(self.info_path)

    def _load(self) -> None:
        """Load the annotation resources

        Args:
            descriptor (str): selected descriptor from one of the annotations
        """
        # Load the field's data
        filename = "data/annotations/slogan_descriptor_combos.json"
        with open(filename, "r") as f:
            self.combos = json.load(f)
            self.info_path = list(self.combos.keys())
        


class RandomSampler:
    def __init__(self,data_source,batch_size):
        self.lengths = [ex[2] for ex in data_source.examples]
        self.batch_size = batch_size

    def randomize(self):
        #random.shuffle(
        N = len(self.lengths)
        self.ind = np.arange(0,len(self.lengths))
        np.random.shuffle(self.ind)
        self.ind = list(self.ind)
        self.ind.sort(key = lambda x: self.lengths[x])
        self.block_ids = {}
        random_block_ids = list(range(N))
        np.random.shuffle(random_block_ids)
        #generate a random number between 0 to N - 1
        blockid = random_block_ids[0]
        self.block_ids[self.ind[0]] = blockid
        running_count = 1 
        for ind_it in range(1,N):
            if running_count >= self.batch_size or self.lengths[self.ind[ind_it]] != self.lengths[self.ind[ind_it-1]]:
                blockid = random_block_ids[ind_it]
                running_count = 0 
            #   
            self.block_ids[self.ind[ind_it]] = blockid
            running_count += 1
        #  
        # Pdb().set_trace()
        self.ind.sort(key = lambda x: self.block_ids[x])
         

    def __iter__(self):
        # Pdb().set_trace()
        self.randomize()
        return iter(self.ind)

    def __len__(self):
        return len(self.ind)

class VQABatchSampler:
    def __init__(self, data_source, batch_size, drop_last=False):
        self.lengths = [ex[2] for ex in data_source.examples]
        # TODO: Use a better sampling strategy.
        # self.sampler = torch.utils.data.sampler.SequentialSampler(data_source)
        self.sampler = RandomSampler(data_source,batch_size)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data_source = data_source
        self.unk_emb = 1000

    def __iter__(self):
        batch = []
        prev_len = -1
        this_batch_counter = 0
        for idx in  self.sampler:
            if self.data_source.examples[idx][4] == self.unk_emb:
                continue
            #
            curr_len = self.lengths[idx]
            if prev_len > 0 and curr_len != prev_len:
                yield batch
                batch = []
                this_batch_counter = 0
            #
            batch.append(idx)
            prev_len = curr_len
            this_batch_counter += 1
            if this_batch_counter == self.batch_size:
                yield batch
                batch = []
                prev_len = -1
                this_batch_counter = 0
        #
        if len(batch) > 0 and not self.drop_last:
            yield batch
            #self.sampler.randomize()
            prev_len = -1
            this_batch_counter = 0

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
