## Software setup

We will use the following python libraries:

1. PyTorch
2. VQA Python API (https://github.com/GT-Vision-Lab/VQA)

Based on Python 3. The VQA API is in Python 2.7; for convenience a version that has been converted to work with Python 3 has been provided. 

## Data Loader 
In this task you will write a dataset loader for VQA (1.0 Real Images, Open-Ended). You should look over the 
original VQA paper[1] to get an idea for this dataset and the task you're about to do.

More specifically your goal for this task is to implement a subclass of torch.utils.data.Dataset (https://pytorch.org/docs/stable/data.html) 
to provide easy access to the VQA data for the latter parts of this assignment. 

The full dataset itself is quite large (the COCO training images are ~13 GB, for instance). For convenience we've provided 
an extremely trimmed down version in the test folder (one image and a couple corresponding questions and annotations) so 
you don't need to download the full dataset on your local machine. The test_vqa_dataset.py file in the test folder has a 
couple simple unit tests that call your implementation using this small dataset. These can be run using `python -m unittest discover test`.

The unit tests are quite easy, just a sanity check. The real test will come as you start using it to train your nets. Don't worry if 
you find yourself coming back to this implementation and improving it as you progress through the assignment.

###Fill in the vqa_dataset.py stub
1. The function \_\_getitem\_\_ should return a dictionary of Tensors. You'll probably want to reference the papers listed at the bottom 
of this document; make your design as reusable as possible (ie it should be usable for both the Simple and the Co-Attention implementations).
1. The function \_\_len\_\_ should return the number of entries in the dataset.
1. Feel free to add arguments and methods as necessary to the stub; what's there is just to get you started.
1. You should use the VQA python API (https://github.com/GT-Vision-Lab/VQA). As it is in Python 2.7, we've provided a Python 3 version 
in the external folder.
1. You may want to review the VQA lecture slides on vocabulary. Consider what the vocabulary for the test set should be.
1. When you are done the unit tests should pass. (Feel free to add assertions to the tests if you find that helpful.)

**Q1** Give a summary of your dataset implementation. What should be in the \_\_getitem\_\_ dictionary? 
How should that information be encoded as Tensors? If you consider multiple options, explain your thought process in picking. What preprocessing should be done? 

###Download the full dataset

For this assignment, you will ultimately need to use the full dataset, so you'll need to download the train and
validation data: https://visualqa.org/vqa_v1_download.html

1. You'll need to get all three things: the annotations, the questions, and the images for both the training and validation sets.
    1. We're just using the validation set for testing, for simplicity. (In other words, we're not creating a separate set for parameter tuning.)
1. You'll need at least 20 GB of space. If you're using AWS Volumes I suggest getting a volume with at least 50 GB for caching (more details in Task 3 below).
1. We're using VQA v1.0 Open-Ended for easy comparison to the baseline papers. Feel free to experiment with, for example, VQA 2.0 [4] if you feel inclined, though it is not required. 

## Simple Baseline
For this task you will implement the simple method described in [2]. This serves to validate your dataset and provide
a baseline to compare the future tasks against.

We've provided a skeleton to get you started. Here's a quick overview:

1. The entry point is main.py, which can be run with `python -m student_code.main` plus any arguments (refer to main.py for the list).
1. Main will, by default, create a SimpleBaselineExperimentRunner.py, which subclasses ExperimentRunnerBase.
1. ExperimentRunnerBase runs the training loop and validation.
1. SimpleBaselineExperimentRunner is responsible for creating the datasets, creating the model (SimpleBaselineNet), and running optimization.
    1. In general anything SimpleBaseline-specific should be put in here, not in the base class.
    
To implement Simple Baseline you'll need to:

1. Create the network in simple_baseline_net.py according to the paper
1. Fill in the missing parts of ExperimentRunnerBase:
    1. Call self._model using your tensors from your dataset from Part 1 (in train())
    1. Extract the ground truth answer for use in _optimize() (in train())
    1. Add code to graph your loss and validation accuracies as you train (in train())
        1. You may use either Visdom or TensorboardX, up to you.
    1. Write the validation loop using _val_dataset_loader (in validate())
1. Implement _optimize() in SimpleBaselineExperimentRunner.
    1. Note that the Simple Baseline paper uses different learning rates for different parts of the network. Consider how this can be handled cleanly in PyTorch.
    
You need not stick strictly to the established structure. Feel free to make changes as desired.

Be mindful of your AWS usage if applicable. If you find yourself using too much, you may wish to use a subset of the dataset for debugging, 
for instance a particular question type (e.g "what color is the"). 

Feel free to refer to the official implementation in Torch (https://github.com/metalbubble/VQAbaseline), 
for instance to find the parameters they used to avoid needing to do a comprehensive parameter search.

While Googlenet will be in future version of torchvision.models, as of this writing that has not yet been pushed to release. So, for convenience, 
googlenet.py has been provided in the external folder.

## Co-Attention Network
In this task you'll implement [3]. This paper introduces three things not used in the Simple Baseline paper: hierarchical question processing, attention, and 
the use of recurrent layers. You may choose to do either parallel or alternating co-attention (or both, if you're feeling inspired).

The paper explains attention fairly thoroughly, so we encourage you to, in particular, closely read through section 3.3 of the paper.

To implement the Co-Attention Network you'll need to:

1. Implement CoattentionExperimentRunner's optimize method. 
1. Implement CoattentionNet
    1. Encode the image in a way that maintains some spatial awareness (see recommendation 1 below).
    1. Implement the hierarchical language embedding (words, phrases, question)
        1. Hint: All three layers of the hierarchy will still have a sequence length identical to the original sequence length. 
        This is necessary for attention, though it may be unintuitive for the question encoding.
    1. Implement your selected co-attention method
    1. Attend to each layer of the hierarchy, creating an attended image and question feature for each
    1. Combine these features to predict the final answer

You may find the implementation of the network to be nontrivial. Here are some recommendations:

1. Pay attention to the image encoding; you may want to skim through [5] to get a sense for why they upscale the images.
1. Consider the attention mechanism separately from the hierarchical question embedding. In particular, you may consider writing 
a separate nn.Module that handles only attention (e.g some "AttentionNet"), that the CoattentionNet can then use.
1. Review the ablation section of the paper (4.4). You can see that you can get good results using only simpler subsets of the 
larger network. You can use this fact to test small subnets (e.g images alone, without any question hierarchy at all), then 
slowly build up the network while making sure that training is still proceeding effectively.
1. The paper uses a batch_size of 300, which we recommend using if you can. One way you can make this work is to pre-compute 
the pretrained network's (e.g ResNet) encodings of your images and cache them, and then load those instead of the full images. This reduces the amount of 
data you need to pull into memory, and greatly increases the size of batches you can run.
    1. This is why we recommended you create a larger AWS Volume, so you have a convenient place to store this cache.


## Relevant papers:
[1] VQA: Visual Question Answering (Agrawal et al, 2016): https://arxiv.org/pdf/1505.00468v6.pdf

[2] Simple Baseline for Visual Question Answering (Zhou et al, 2015): https://arxiv.org/pdf/1512.02167.pdf

[3] Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017):  https://arxiv.org/pdf/1606.00061.pdf

[4] Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering (Goyal, Khot et al, 2017):  https://arxiv.org/pdf/1612.00837.pdf

[5] Stacked Attention Networks for Image Question Answering (Yang et al, 2016): https://arxiv.org/pdf/1511.02274.pdf
