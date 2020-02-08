# Image Caption Baseline

Requirements: Pytorch 1.2 and python 3.7 

Structure: CNN+LSTM+Attention

## Implementation Details

### Encoder

We use a pretrained ResNet-101 already available in PyTorch's `torchvision` module and discard the last two layers (pooling and linear layers).  

We do add an `AdaptiveAvgPool2d()` layer to **resize the encoding to a fixed size**. This makes it possible to feed images of variable size to the Encoder. (We resize our input images to `256, 256` because we had to store them together as a single tensor.) 

We **only fine-tune convolutional blocks 2 through 4 in the ResNet**, because the first convolutional block would have usually learned something very fundamental to image processing, such as detecting lines, edges, curves, etc.  

### Attention

Separate linear layers **transform both the encoded image (flattened to `N, 14 * 14, 2048`, N represents the batch size) and the hidden state (output) from the Decoder to the same dimension**, viz. the Attention size. They are then added and ReLU activated. A third linear layer **transforms this result to a dimension of 1**, whereupon we **apply the softmax to generate the weights** `alpha`.

### Decoder

The output of the Encoder is received here and flattened to dimensions `N, 14 * 14, 2048`. This is just convenient and prevents having to reshape the tensor multiple times.

We **initialize the hidden and cell state of the LSTM** using the encoded image with the `init_hidden_state()` method, which uses two separate linear layers.

At the very outset, we **sort the `N` images and captions by decreasing caption lengths**. This is so that we can process only _valid_ timesteps, i.e., not process the `<pad>`s. 

The **iteration is performed _manually_ in a `for` loop** with a PyTorch [`LSTMCell`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM) instead of iterating automatically without a loop with a PyTorch [`LSTM`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM). This is because we need to execute the Attention mechanism between each decode step. An `LSTMCell` is a single timestep operation, whereas an `LSTM` would iterate over multiple timesteps continously and provide all outputs at once. 

We **compute the weights and attention-weighted encoding** at each timestep with the Attention network. In section `4.2.1` of the paper, they recommend **passing the attention-weighted encoding through a filter or gate**. This gate is a sigmoid activated linear transform of the Decoder's previous hidden state. The authors state that this helps the Attention network put more emphasis on the objects in the image. 

We **concatenate this filtered attention-weighted encoding with the embedding of the previous word** (`<start>` to begin), and run the `LSTMCell` to **generate the new hidden state (or output)**. A linear layer **transforms this new hidden state into scores for each word in the vocabulary**, which is stored.

We also store the weights returned by the Attention network at each timestep.  

## Training Model 

Before you begin, make sure to save the required data files for training, validation, and testing. To do this, run the contents of [`create_input_files.py`] after pointing it to the the Karpathy JSON file and the image folder containing the extracted `train2014` and `val2014` folders from your [downloaded data]. 

I'm using the MSCOCO '14 Dataset. You'd need to download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images.

We will use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contain the captions. You will also find splits and captions for the Flicker8k and Flicker30k datasets, so feel free to use these instead of MSCOCO if the latter is too large for computer. 

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py).
The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you wish to.

To **train your model from scratch**, simply run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.
Note that we perform validation at the end of every training epoch.

## Reference

[1] https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning 

[2] https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
