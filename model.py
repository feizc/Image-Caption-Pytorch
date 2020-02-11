import torch
from torch import nn
import torchvision

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
	"""Image Encoder"""
	def __init__(self, encode_image_size=14):
		super(Encoder, self).__init__()
		self.encode_image_size = encode_image_size
		
		#pretrained res-net 101
		resnet=torchvision.models.resnet101(pretrained=True)

		#remove last two layer (linear and pool)
		modules=list(resnet.children())[:-2]
		self.resnet=nn.Sequential(*modules)

		#resize image feature vectors 14*14
		self.adaptive_pool=nn.AdaptiveAvgPool2d((encode_image_size,encode_image_size))

		self.fine_tune()

	def forward(self, images):
		"""
		param: (batch_size, 3, H, W)
		return: image feature vector
		"""

		#(batch_size, channel(2048), H/32, W/32)
		out=self.resnet(images)
		#(batch_size, 2048, 14, 14)
		out=self.adaptive_pool(out)
		#(batch_size, 14, 14, 2048)
		out=out.permute(0,2,3,1)
		return out

	def fine_tune(self,fine_tune=True):
		"""
		fine-tune high-level layers
		"""
		for p in self.resnet.parameters():
			p.requires_grad=False
		for c in list(self.resnet.children())[5:]:
			for p in c.parameters():
				p.requires_grad=fine_tune

class Attention(nn.Module):
	"""Attention"""
	def __init__(self, encoder_dim, decoder_dim, attention_dim):
		super(Attention, self).__init__()
		self.encoder_attention=nn.Linear(encoder_dim, attention_dim)
		self.decoder_attention=nn.Linear(decoder_dim, attention_dim)
		self.full_attention=nn.Linear(attention_dim,1)
		self.relu=nn.ReLU()
		self.softmax=nn.Softmax(dim=1)

	def forward(self, encoder_out, decoder_hidden):
		"""
		param: encoder_out (batch_size, num_pixels=14*14, encoder_dim=2048)
			   decoder_hidden (batch_size, decoder_dim)
		return: attention weighted feature vector, weights
		"""

		#(batch_size, num_pixels, attention_dim)
		att1=self.encoder_attention(encoder_out)
		#(batch_size, attention_dim)
		att2=self.decoder_attention(decoder_hidden)
		#(batch_size, num_pixels)
		att=self.full_attention(self.relu(att1+att2.unsqueeze(1))).squeeze(2)
		alpha=self.softmax(att)
		#(batch_size, encoder_dim)
		attention_weighted_encoding=(encoder_out*alpha.unsqueeze(2)).sum(dim=1)

		return attention_weighted_encoding, alpha


class Decoder(nn.Module):
	"""Decoder"""
	def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
		#deocoder_dim: size of decoder's RNN
		super(Decoder, self).__init__()
		self.attention_dim=attention_dim
		self.embed_dim=embed_dim
		self.decoder_dim=decoder_dim
		self.vocab_size=vocab_size
		self.encoder_dim=encoder_dim
		self.dropout=dropout

		self.attention=Attention(encoder_dim, decoder_dim, attention_dim)
		self.embedding=nn.Embedding(vocab_size, embed_dim)
		self.dropout=nn.Dropout(p=self.dropout)
		self.decode_step=nn.LSTMCell(embed_dim+encoder_dim, decoder_dim, bias=True)
		self.init_h_0=nn.Linear(encoder_dim, decoder_dim)
		self.init_c_0=nn.Linear(encoder_dim, decoder_dim)
		self.f_beta=nn.Linear(decoder_dim, encoder_dim) #sigmoid-activated gate
		self.sigmoid=nn.Sigmoid()
		self.fc=nn.Linear(decoder_dim, vocab_size)
		self.init_weights()

	def init_weights(self):
		"""
		Initilize parameter values with uniform distribution
		"""
		self.embedding.weight.data.uniform_(-0.1,0.1)
		self.fc.bias.data.fill_(0)
		self.fc.weight.data.uniform_(-0.1,0.1)

	def load_pretrained_embeddings(self,embeddings):
		"""
		Load embedding layer with pre-trained embeddings
		"""
		self.embedding.weight=nn.Parameter(embeddings)
		
	def fine_tune_embedding(self, fine_tune=True):
		for p in self.embedding.parameters():
			p.requires_grad=fine_tune

	def init_hidden_state(self, encoder_out):
		"""
		param: encoder_out (batch_size, num_pixels, decoder_dim)
		return: h_0, c_0
		"""
		mean_encoder_out=encoder_out.mean(dim=1)
		#(batch_size, decoder_dim)
		h_0=self.init_h_0(mean_encoder_out)
		c_0=self.init_c_0(mean_encoder_out)
		return h_0, c_0

	def forward(self, encoder_out, encoded_captions, caption_lengths):
		"""
		param: encoder_out (batch_size, 14, 14, encoder_dim=2048)
		       encoded_caption (batch_size, max_caption_length)
		       caption_length (batch_size, 1)
		return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
		"""
		batch_size=encoder_out.size(0)
		encoder_dim=encoder_out.size(2)
		vocab_size=self.vocab_size

		#flatten image (batch_size, num_pixels, encoder_dim)
		encoder_out=encoder_out.view(batch_size, -1, encoder_dim)
		num_pixels=encoder_out.size(1)

		#sort input data by decresing lengths
		caption_lengths, sort_idx=caption_lengths.squeeze(1).sort(dim=0, descending=True)
		encoder_out=encoder_out[sort_idx]
		encoded_caption=encoded_captions[sort_idx]

		# (batch_size, max_caption_length, embed_dim)
		embeddings=self.embedding(encoded_caption)
		# (batch_size, decoder_dim)
		h, c=self.init_hidden_state(encoder_out)

		#since we don't decode <eos> position, decoding length=actual lengths-1
		decode_lengths=(caption_lengths-1).tolist()

		predictions=torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
		alphas=torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

		#at each time step, generate a new word based on the previous word and the attention weighted encoding
		for t in range(max(decode_lengths)):
			batch_size_t=sum([l>t for l in decode_lengths])
			attention_weighted_encoding, alpha=self.attention(encoder_out[:batch_size_t],
															  h[:batch_size_t])
			# (batch_size_t, encoder_dim)
			gate=self.sigmoid(self.f_beta(h[:batch_size_t]))
			attention_weighted_encoding=gate*attention_weighted_encoding
			h, c=self.decode_step(
				torch.cat([embeddings[:batch_size_t,t,:],attention_weighted_encoding],dim=1),
				(h[:batch_size_t],c[:batch_size_t]))
			# (batch_size_t, vocab_size)
			preds=self.fc(self.dropout(h))
			predictions[:batch_size_t,t,:]=preds
			alphas[:batch_size_t,t,:]=alpha

		return predictions, encoded_captions, decode_lengths, alphas, sort_idx








		