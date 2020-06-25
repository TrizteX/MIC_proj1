from neural_style import *

content_img, style_img = ('images/content1.jpg','images/style1.jpg')

nst=Neural_Style(num_steps=300)

output = nst.run_style_transfer(content_img, style_img)

nst.imshow(output)
