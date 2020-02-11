import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform 
import argparse #read cmd param
from scipy.misc import imread, imresize #io for image process
from PIL import Image 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_beam_search(encoder, decoder, image_path, word_map, beam_size=5):
	k=beam_size
	vocab_size=len(word_map)

	#read image and process
	img=imread(image_path)
	if len(img.shape)==2:
		img=img[:,:,np.newaxis]
		img=np.concatenate([img,img,img],axis=2)
	img=imresize(img,(256,256))
	#(3,256,256)
	img=img.transpose(2,0,1)
	img=img/256
	img=torch.FloatTensor(img).to(device)
	normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])
	tranform=transforms.Compose([normalize])
	img=transform(img)

	#encode image
	#(1,3,256,256)
	image=image.unsqueeze(0)
	#(1,14,14,2048)
	encoder_out=encoder(image)
	encoded_image_size=encoder_out.size(1)
	encoder_dim=encoder_out.size(3)
	#(1,14*14,2048)
	encoder_out=encoder_out.view(1,-1,encoder_dim)
	num_pixels=encoder_out.size(1)

	#treat the problem as having a batch size of k
	encoder_out=encoder_out.expend(k, num_pixels, encoder_dim)
	#store top k previous words at each step (k,1)
	k_prev_words=torch.LongTensor([[word_map['<start>']]]*k).to(device)
	#store top k sequences
	seqs=k_prev_words
	#store top k sequences' scores
	top_k_scores=torch.zeros(k,1).to(device)
	#store top k sequences' alphas (k,1,14,14)
	seqs_alpha=torch.ones(k,1,encoded_image_size, encoded_image_size).to(device)
	complete_seqs=list()
	complete_seq_alpha=list()
	complete_seq_scores=list()

	#caption decoding
	step=1
	h,c=decoder.init_hidden_state(encoder_out)

	while True:
		# (s, embed_dim) s is a number less or equal to k
		embeddings=decoder.embedding(k_prev_words).squeeze(1)
		# (s, encoder_dim), (s, num_pixels)
		awe, alpha=decoder.attention(encoder_out,h)
		# (s, 14, 14)
		alpha=alpha.view(-1, encoded_image_size, encoded_image_size)
		# (s, encoder_dim)
		gate=decoder.sigmoid(decoder.f_beta(h))
		awe=gate*awe
		# (s, decoder_dim)
		h,c=decoder.decode_step(torch.cat([embeddings,awe],dim=1),(h,c))
		# (s, vocab_size)
		scores=decoder.fc(h)
		scores=F.log_softmax(scores, dim=1)
		scores=top_k_scores.expand_as(scores)+scores

		if step==1:
			top_k_scores, top_k_words=scores[0].topk(k,dim=0, largest=True, sorted=True)
		else:
			top_k_scores, top_k_words=scores.view(-1).topk(k, dim=0,largest=True, sorted=True)

		#convert unroll indices to actual indices of scores
		prev_word_idx=top_k_words/vocab_size
		next_word_idx=top_k_words%vocab_size

		# (s, step+1)
		seqs=torch.cat([seqs[prev_word_idx],next_word_idx.unsqueeze(1)],dim=1)
		# (s, step+1, 14,14)
		seqs_alpha=torch.cat([seqs_alpha[prev_word_idx], alpha[prev_word_idx].unsqueeze(1)], dim=1)

		incomplete_idx=[idx for idx, next_word in enumerate(next_word_idx) if next_word!=word_map['<end>']]
		complete_idx=list(set(range(len(next_word_idx)))-set(incomplete_idx))

		# set aside complete sequences
		if len(complete_idx)>0:
			complete_seqs.extend(seqs[complete_idx].tolist())
			complete_seq_alpha.extend(seqs_alpha[complete_idx].tolist())
			complete_seq_scores.extend(top_k_scores[complete_idx])
		k-=len(complete_idx)
		if k==0:
			break

		#process incomplete sequences
		seqs=seqs[incomplete_idx]
		seqs_alpha=seqs_alpha[incomplete_idx]
		h=h[prev_word_idx[incomplete_idx]]
		c=c[prev_word_idx[incomplete_idx]]
		encoder_out=encoder_out[prev_word_idx[incomplete_idx]]
		top_k_scores=top_k_scores[incomplete_idx].unsqueeze(1)
		k_prev_words=next_word_idx[incomplete_idx].unsqueeze(1)

		if step>20:
			break
		step+=1

	i=complete_seq_scores.index(max(complete_seq_scores))
	seq=complete_seqs[i]
	alphas=complete_seq_alpha[i]
	return seq, alphas


def attention_visualize(image_path, seq, alphas, rev_word_map, smooth=True):
	"""
	param: rev_word_map: reverse word mapping, idx2word
	"""
	image=Image.open(image_path)
	image=image.resize([14*24, 14*24], Image.LANCZOS)
	words=[rev_word_map[idx] for idx in seq]

	for t in range(len(words)):
		if t>20:
			break
		plt.subplot(np.ceil(len(words)/5.),5,t+1)
		plt.text(0,1,'%s'%(words[t]), color='black',backgroundcolor='white',fontsize=12)
		plt.imshow(image)
		current_alpha=alphas[t,:]
		if smooth:
			alpha=skimage.tranform.pyramid_expand(current_alpha.numpy(),upscale=24,sigma=8)
		else:
			alpha=skimage.tranform.resize(current_alpha.numpy(),[14*24,14*24])
		if t==0:
			plt.imshow(alpha,alpha=0)
		else:
			plt.imshow(alpha,alpha=0.8)
		plt.set_cmap(cm.Greys_r)
		plt.axis('off')
	plt.show()


if __name__ == '__main__':
	parser=argparse.ArgumentParser(description='image caption baseline')

	parser.add_argument('--img','-i',help='path to image')
	parser.add_argument('--model','-m', help='path to model')
	parser.add_argument('--word_map','-wm',help='path to word map JSON')
	parser.add_argument('--beam_size','-b',default=5, type=int, help='beam size for beam search')
	parser,add_argument('--dont_smooth', dest='smooth',
						action='store_false', help='do not smooth alpha overlay')

	args=parser.parse_args()

	#load model
	checkpoint=torch.load(args.model)
	decoder=checkpoint['decoder']
	decoder=decoder.to(device)
	decoder.eval()
	encoder=checkpoint['encoder']
	encoder=encoder.to(device)
	encoder.eval()

	#load world map
	with open(args.word_map,'r') as j:
		word_map=json.load(j)
	rev_word_map={v:k for k,v in word_map.items()}

	#model running
	seq, alphas=caption_beam_search(encoder, decoder, args.img, word_map, args.beam_size)
	alphas=torch.FloatTensor(alphas)

	attention_visualize(args.img, seq, alphas, rev_word_map, args.smooth)














