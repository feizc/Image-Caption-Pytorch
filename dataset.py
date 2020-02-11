import torch
from torch.utils.data import Dataset
import h5py
import json
import os

class CaptionDataset(Dataset):
	"""CaptionDataset Class"""
	def __init__(self, data_folder, data_name, split, transform=None):
		"""
		param: split (TRAIN, VAL or TEST)
		"""
		super(CaptionDataset, self).__init__()
		self.split=split
		assert self.split in {'TRAIN','VAL','TEST'}

		#open hdf5 file where image stored
		self.h=h5py.File(os.path.join(data_folder, self.split+'_IMAGES_'+data_name+'.hdf5'),'r')
		self.imgs=self.h['images']
		#caption_per_image
		self.cpi=self.h.attrs['captions_per_image']

		#load encoded captions into memory
		with open(os.path.join(data_folder,self.split+'_CAPTIONS_'+data_name+'.json'),'r') as j:
			self.captions=json.load(j)

		#load caption length in to memory
		with open(os.path.join(data_folder,self.split+'_CAPLENS_'+data_name+'.json'),'r') as j:
			self.caplens=json.load(j)

		self.transform=transform
		#total number of captions
		self.dataset_size=len(self.captions)

	def __getitem__(self,i):
		#the Nth caption corresponds to (N//caption_per_image)th image
		img=torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
		if self.transform is not None:
			img=self.transform(img)
		caption=torch.LongTensor(self.captions[i])
		caplen=torch.LongTensor([self.caplens[i]])

		if self.split is 'TRAIN':
			return img, caption, caplen
		else:
			#return all captions to calculate BLEU-4 score
			all_captions=torch.LongTensor(self.captions[((i//self.cpi)*self.cpi):(((i//self.cpi)*self.cpi)+self.cpi)])
			return img, caption, caplen, all_captions
			
	def __len__(self):
		return self.dataset_size



