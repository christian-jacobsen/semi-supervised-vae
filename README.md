# Semi-Supervised VAE

In the literature, semi-supervised regression tasks seem far less explored than semi-supervised classification tasks. Additionally, when combining semi-supervised approaches with generative modelling techniques such as VAEs and GANs, typical labels are for classification of data rather than on the lower dimensional representation itself. Incorporating a few labeled samples into training a VAE could potentially aid in learning disentangled representations. But how do we incorporate labeled data?

Suppose we have high dimensional data from some real world system, from the simulation of some partial differential equation (PDE), or a combination of the two. The data obtained via PDE simulation may be completely labeled, partially labeled, or unlabeled. We can usually assume that the PDE model is not a compeletely accurate representation of the physical world; thus, labeled PDE data can be assumed partially labeled and not fully labeled. Some factors of variation included in the model also may have been constant in PDE simulations but are not in the real world.

However, if we do have a combination of labeled (for now we will assume fully labeled -- this is usually not the case) and unlabeled data, how can we take advantage of the labeled data to learn correct, and therefore, disentangled representations? 

## Setup

Suppose that some system exists 
