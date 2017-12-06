# Simple version of spatial_transformer.py, work on both channels  
import numpy as np 
import cv2 
import numpy as np 
import cv2 
import pdb 
import matplotlib.pyplot as plt 
from skimage import io 
import torch 
import torch.nn.functional as F 
###############################################################
# Changable parameter
SCALE_H = True 
# scale_H:# The indices of the grid of the target output is
# scaled to [-1, 1]. Set False to stay in normal mode 
def _meshgrid(height, width, scale_H = SCALE_H):
    if scale_H:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                        np.linspace(-1, 1, height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

    else:
        x_t, y_t = np.meshgrid(range(0,width), range(0,height))
        ones = np.ones(np.prod(x_t.shape))
        grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # print '--grid size:', grid.shape 
    return grid 

def _repeat(x, n_repeats):
    """x: (N, )"""
 
    rep = np.transpose(np.ones([n_repeats, 1]), [1, 0] ).astype(np.int32) # 1 x out_width*out_height 
    x = np.dot(x.reshape([-1, 1]), rep) # N x 1   x  1 x out_width*out_height = N   x   out_width*out_height
    return x.reshape([-1])

def _interpolate(im, x, y, out_size, scale_H = SCALE_H):
    """im: N x d x H x W
        x: N x D 
        y: N x D"""
    # constants
    batch_size, num_channels, height, width = im.shape 

    height_f = float(height)
    width_f =  float(width)
    out_height = out_size[0]
    out_width = out_size[1]
    zero = np.zeros([], dtype='int32')
    max_y = height - 1
    max_x = width - 1

    if scale_H:
        # # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

    # do sampling
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

 

    # print 'x0:', y0 
    # print 'x1:', y1 
    # Limit the size of the output image 
    x0 = np.clip(x0, zero, max_x)
    x1 = np.clip(x1, zero, max_x)
    y0 = np.clip(y0, zero, max_y)
    y1 = np.clip(y1, zero, max_y)
    

    # To extract the images, let's first latten batched image im, then gather and reshape later 
    dim2 = width 
    dim1 = width*height 
    base = _repeat(np.arange(batch_size)*dim1, out_height*out_width)
    base_y0 = base + y0*dim2 
    base_y1 = base + y1*dim2 

    idx_a = base_y0 + x0 
    idx_b = base_y1 + x0 
    idx_c = base_y0 + x1 
    idx_d = base_y1 + x1 

    im_flat = im.reshape([-1, num_channels])
    im_flat = im_flat.astype(np.float32)

    Ia = np.take(im_flat, idx_a, axis=0) 
    Ib = np.take(im_flat, idx_b, axis=0) 
    Ic = np.take(im_flat, idx_c, axis=0) 
    Id = np.take(im_flat, idx_d, axis=0) 

    x0_f = x0.astype(np.float32)
    x1_f = x1.astype(np.float32)
    y0_f = y0.astype(np.float32)
    y1_f = y1.astype(np.float32)

    wa = np.expand_dims(((x1_f-x) * (y1_f-y)), 1)
    wb = np.expand_dims(((x1_f-x) * (y-y0_f)), 1)
    wc = np.expand_dims(((x-x0_f) * (y1_f-y)), 1)
    wd = np.expand_dims(((x-x0_f) * (y-y0_f)), 1)
    
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    # print '--shape of out:', out.shape
    return out 

def _transform(theta, input_dim, out_size):
    """Theta: N x 6 or N x 9 
       input_dim: N x d x H x W"""
    batch_size, num_channels, height, width = input_dim.shape 
    tf_mat_numel = theta.shape[1]
    if tf_mat_numel == 6:
        is_affine = True 
    else:
        is_affine = False 
    if is_affine:
        theta = np.reshape(theta, (-1, 2, 3))
    else:
        theta = np.reshape(theta, (-1, 3, 3))
 
 
    print('-- Theta shape:', theta.shape) 

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = out_size[0]
    out_width = out_size[1]
    grid = _meshgrid(out_height, out_width) # 3 x D 

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = np.dot(theta, grid)
    x_s = T_g[:,0,:]
    y_s = T_g[:,1,:]
    t_s = T_g[:,2,:]
    # print '-- T_g:', T_g 
    # print '-- x_s:', x_s 
    # print '-- y_s:', y_s
    # print '-- t_s:', t_s

    t_s_flat = np.reshape(t_s, [-1])
    # Ty changed 
    if is_affine:
        x_s_flat = np.reshape(x_s, [ -1])
        y_s_flat = np.reshape(y_s, [-1])
    else:
        x_s_flat = np.reshape(x_s, [-1])/t_s_flat
        y_s_flat = np.reshape(y_s, [-1])/t_s_flat
    

    input_transformed =  _interpolate(input_dim, x_s_flat, y_s_flat, out_size) 
   
    output = np.reshape(input_transformed, [batch_size,num_channels, out_height, out_width]).astype(np.uint8)
 
    return output


def numpy_transformer(img, H, out_size, scale_H = SCALE_H): 
    h, w = img.shape[0], img.shape[1]
    # Matrix M 
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

    if scale_H:
        H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
        # print 'H_transformed:', H_transformed 
        img2 = _transform(H_transformed, img, [h,w])
    else:
        img2 = _transform(np.linalg.inv(H), img, [h,w])
    return img2 

def test_torch_spatial_transformer():
    img = io.imread('/home/tynguyen/cis680/data/cifar10_transformed/imgs/05975.png')
    h, w = img.shape[0], img.shape[1]
    print( '-- h, w:', h, w ) 


    # Apply homography transformation 

    H = np.array([[2., 0.3, 5], [0.3, 2., 10.], [0, 0, 1]]).astype(np.float32)
    img2 = cv2.warpAffine(img, H[:2, :], (w, h))
    
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)


    H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
   
    print('H_transformed:', H_transformed)  
    
    img_tensor = torch.unsqueeze(torch.from_numpy(img).float(), 0) 
    img_tensor = img_tensor.permute(0, 3, 1, 2) 
    affine_tensor = torch.unsqueeze(torch.from_numpy(H_transformed[:2,:]).float(), 0)
    pdb.set_trace()
    grid = F.affine_grid(affine_tensor, img_tensor.size())
    img3 = F.grid_sample(img_tensor, grid).data.numpy()[0].astype(np.uint8)  
    img3 = img3.transpose(1,2,0)  
    
    print ( '-- Reprojection error:', np.mean(np.abs(img3 - img2))) 
    Reprojection = abs(img3 - img2)
    # Test on real image 
    count = 0 
    amount = 0 
    #for i in range(48):
    #  for j in range(48):
    #    for k in range(2):
    #      if Reprojection[i, j, k] > 10:
    #        print(i, j, k, 'value', Reprojection[i, j, k])
    #        count += 1 
    #        amount += Reprojection[i, j, k]
    #print('There is total %d > 10, over total %d, account for %.3f'%( count, 48*48*3,amount*1.0/count) ) 
    
    #io.imshow('img3', img3) 
    try:
        plt.subplot(221)
        plt.imshow(img)
        plt.title('Original image')

        plt.subplot(222)
        plt.imshow(img2)
        plt.title('cv2.warpPerspective')

        plt.subplot(223)
        plt.imshow(img3)
        plt.title('Transformer')

        plt.subplot(224)
        plt.imshow(Reprojection)
        plt.title('Reprojection Error')
        plt.show()
    except KeyboardInterrupt:
        plt.close()
        exit(1)



def test_transformer(scale_H = SCALE_H): 
    img = io.imread('/home/tynguyen/cis680/data/cifar10_transformed/imgs/05975.png')
    h, w = img.shape[0], img.shape[1]
    print( '-- h, w:', h, w ) 


    # Apply homography transformation 

    H = np.array([[2., 0.3, 5], [0.3, 2., 10.], [0.0001, 0.0002, 1.]]).astype(np.float32)
    img2 = cv2.warpPerspective(img, H, (w, h))


    # # Matrix M 
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

    if scale_H:
        H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
        print('H_transformed:', H_transformed)
        img3 = _transform(H_transformed, img, [h,w])
    else:
        img3 = _transform(np.linalg.inv(H), img, [h,w])

    print ( '-- Reprojection error:', np.mean(np.abs(img3 - img2))) 
    Reprojection = abs(img3 - img2)
    # Test on real image 
    count = 0 
    amount = 0 
    for i in range(48):
      for j in range(48):
        for k in range(2):
          if Reprojection[i, j, k] > 10:
            print(i, j, k, 'value', Reprojection[i, j, k])
            count += 1 
            amount += Reprojection[i, j, k]
    print('There is total %d > 10, over total %d, account for %.3f'%( count, 48*48*3,amount*1.0/count) ) 
    
    #io.imshow('img3', img3) 
    try:
        plt.subplot(221)
        plt.imshow(img)
        plt.title('Original image')

        plt.subplot(222)
        plt.imshow(img2)
        plt.title('cv2.warpPerspective')

        plt.subplot(223)
        plt.imshow(img3)
        plt.title('Transformer')

        plt.subplot(224)
        plt.imshow(Reprojection)
        plt.title('Reprojection Error')
        plt.show()
    except KeyboardInterrupt:
        plt.close()
        exit(1)



def test_batch_transformer(scale_H = SCALE_H): 
    img1 = io.imread('/home/tynguyen/cis680/data/cifar10_transformed/imgs/05975.png')
    img2 = io.imread('/home/tynguyen/cis680/data/cifar10_transformed/imgs/05974.png')
    img_batch = np.vstack((np.expand_dims(img1, 0), np.expand_dims(img2,0)))
    img_batch = np.transpose(img_batch, (0, 3, 1, 2)) # Permute to N x d x H x W 
    h, w = img1.shape[0], img1.shape[1]
    print( '-- h, w:', h, w ) 


    # Apply homography transformation 

    H = np.array([[2., 0.3, 5], [0.3, 2., 10.], [0.0001, 0.0002, 1.]]).astype(np.float32)
    w_img1 = cv2.warpPerspective(img1, H, (w, h))


    # # Matrix M 
    M = np.array([[w/2.0, 0, w/2.0], [0, h/2.0, h/2.0], [0, 0, 1.]]).astype(np.float32)

    if scale_H:
        H_transformed = np.dot(np.dot(np.linalg.inv(M), np.linalg.inv(H)), M)
        H_batch = np.tile(np.expand_dims(H_transformed.flatten(), 0), [2, 1]) 

      
        tf_img = _transform(H_batch, img_batch, [h,w])
        tf_img1 = tf_img[0]
        tf_img1 = np.transpose(tf_img1, (1, 2, 0)).astype(np.uint8)
    else:
        tf_img1 = _transform(np.linalg.inv(H), img1, [h,w])

    print ( '-- Reprojection error:', np.mean(np.abs(w_img1 - tf_img1))) 
    Reprojection = abs(w_img1 - tf_img1)
    # Test on real image 
    count = 0 
    amount = 0 
    # for i in range(48):
    #   for j in range(48):
    #     for k in range(2):
    #       if Reprojection[i, j, k] > 10:
    #         print(i, j, k, 'value', Reprojection[i, j, k])
    #         count += 1 
    #         amount += Reprojection[i, j, k]
    # print('There is total %d > 10, over total %d, account for %.3f'%( count, 48*48*3,amount*1.0/count) ) 
    
    #io.imshow('tf_img1', tf_img1) 
    try:
        plt.subplot(221)
        plt.imshow(img1)
        plt.title('Original image')

        plt.subplot(222)
        plt.imshow(w_img1)
        plt.title('cv2.warpPerspective')

        plt.subplot(223)
        plt.imshow(tf_img1)
        plt.title('Transformer')

        plt.subplot(224)
        plt.imshow(Reprojection)
        plt.title('Reprojection Error')
        plt.show()
    except KeyboardInterrupt:
        plt.close()
        exit(1)


if __name__ == "__main__":
    #test_transformer()
  test_batch_transformer()
