import torch
import torch.nn as nn


class DiceLoss:
    @staticmethod

    # TODO: dice needs threshold here no, actually no, so the network after reaching 0.5 has no incentive to become "more certain"

    def dice_loss(pred, target, smooth=0, eps=1e-7, return_loss=True):
        """
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """
        print(type(pred), type(target))
        if not torch.is_tensor(pred) or not torch.is_tensor(target):
            raise TypeError("Input type is not a torch.Tensor. Got {} and {}".format(type(pred), type(target)))
        if not len(pred.shape) == 5:
            raise ValueError("Invalid input shape, we expect BxCxHxWxD. Got: {}".format(pred.shape))
        if not (pred.shape == target.shape):
            raise ValueError("input and target shapes must be the same. Got: {} and {}".format(pred.shape, target.shape))
        if not pred.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {} and {}".format(pred.device, target.device))

        # have to use contiguous since they may from a torch.view op
        iflat = pred[:, 0].contiguous().view(-1)  # one channel only N x 1 x H x D X W -> N x H x D x W
        tflat = target[:, 0].contiguous().view(-1)

        intersection = torch.sum(iflat * tflat)
        A_sum_sq = torch.sum(iflat * iflat)
        B_sum_sq = torch.sum(tflat * tflat)
        dice = (2.0 * intersection + smooth + eps) / (A_sum_sq + B_sum_sq + eps)
        return 1 - dice if return_loss else dice


if __name__ == "__main__":
    loss = DiceLoss.dice_loss
    import numpy as np

    # a = np.random.rand(6,2)
    for _ in range(1000):
        a = np.ones((6, 1, 64, 64, 32))
        b = np.ones((6, 1, 64, 64, 32))
        a = torch.Tensor(a)
        b = torch.Tensor(b)
        l = loss(a, b)
        print(type(l))
        exit(0)
        if loss(a, b) > 0.8:
            # print(a)
            # print(b)
            pass

r""" 

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

-------------------
 
class DiceLoss(nn.Module):
    
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
     
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N

		return loss 

----------------------------------------------------
def dice_loss(true, logits, eps=1e-7):
    Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


------------------------------

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.utils import one_hot


# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

[docs]
class DiceLoss(nn.Module):
    Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
     N = 5  # num_classes
        loss = kornia.losses.DiceLoss()
        input = torch.randn(1, N, 3, 5, requires_grad=True)
        target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
         output = loss(input, target)
         output.backward()
    

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(torch.tensor(1.) - dice_score)



######################
# functional interface
######################


[docs]
def dice_loss(
        input: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
    rFunction that computes Sørensen-Dice Coefficient loss.

    See :class:`~kornia.losses.DiceLoss` for details.
    
    return DiceLoss()(input, target)

----------------------------

def dice_loss(input,target):

    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz


-----------------------------

def dice_loss(pred, target):
    This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

---------------------------------

 """
