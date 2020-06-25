# MIC Project 1: Contribution to PyVision
## Neural Style Transfer: An implementation of the paper "A Neural Algorithm of Artistic Style".

Link to the paper: https://arxiv.org/pdf/1508.06576.pdf

The idea is to extract the _content_ from one image, the 'content image', and the _style_ or _texture_ from another image, the 'style image', to get a single output which has a combination of the two.

## Here are some of the results
Output Images:

![Output](/output/content1+style6.png)
![](/output/content4+style1.png)
![](/output/content6+style7.png)

## To run:

from neural_style import *

content_img, style_img, input_img=image_loader('path_to_content_image','path_to_style_image')

nst=Neural_Style()

output = nst.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,content_img, style_img, input_img)

imshow(output, title='Output Image')  
