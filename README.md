
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

from neural_style import Neural_Style

content_img, style_img = ('path_to_content_image','path_to_style_image')

nst=Neural_Style(num_steps=300, use_gpu=False)

output = nst.run_style_transfer(style_img, content_img)

nst.imshow(output)


