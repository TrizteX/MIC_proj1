def get_path():
    #please provide the paths to the content and style images
    cnt_img='images/content1.jpg'
    sty_img='images/style6.jpg'
    
    return cnt_img, sty_img

def get_variables():
    
    #if you want to use gpu, please set it to y, else to n
    
    exp_var = {'use_gpu': 'y', 'content_layers_default':['conv_4'], 'style_layers_default': ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'], 'num_steps' : 300, 'style_weight': 1000000, 'content_weight':1}
    
    
    #DEFAULT VALUES:
    #use_gpu='y'
    #content_layers_default = ['conv_4']
    #style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    #num_steps = 300
    #style_weight = 1000000
    #content_weight =1
    
    return exp_var