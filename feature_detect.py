import cv2
import numpy as np 
import matplotlib.pyplot as plt

def feature_detect(img_rgb_int):
    # convert to gray
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = img_gray.astype(float) / 255

    # calculate img grads
    img_grad_x, img_grad_y = img_grads(img_gray)
    
    # get harris feature matrix
    feature_mat = harris_feature(img_grad_x, img_grad_y)
    
    # feature response
    k = 0.05
    corner_response = feature_response(feature_mat, img_gray.shape, k)

    # threshold on R
    thresh_r = 0.0001
    feature_set = feature_select(corner_response, thresh_r = thresh_r)

    # non max surpression
    feature_pos_nms = non_max_surpress(feature_set, feature_num = 250)

    return feature_pos_nms

def feature_select(corner_response, thresh_r = 0.01):
    feature_region = corner_response >= thresh_r
    feature_position = np.where(feature_region)
    feature_r = corner_response[feature_region]

    cv2.imshow('My Image', feature_region.astype(float))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    feature_set = {'position': feature_position, 'response': feature_r}
    return feature_set

def feature_response(feature_mat, size_img, k = 0.05):
    pixel_num = feature_mat.shape[1]
    corner_response_flat = np.empty(pixel_num)

    for idx in range(pixel_num):
        s_x, s_xy, s_y = feature_mat[:,idx]
        m = np.array([[s_x, s_xy],[s_xy, s_y]])
        eig_val, _ = np.linalg.eig(m)
        eig_val_1, eig_val_2 = eig_val
        det_m = eig_val_1 * eig_val_2
        trace_m = eig_val_1 + eig_val_2
        corner_response_flat[idx] =det_m - k * (trace_m**2)

    return corner_response_flat.reshape(size_img)

    

def harris_feature(img_grad_x, img_grad_y):
    # hessian term
    img_grad_x2 = img_grad_x ** 2
    img_grad_y2 = img_grad_y ** 2
    img_grad_xy = img_grad_x * img_grad_y

    # gussian filter
    kernel_size, sigma = 9, 2
    img_grad_x2_blur = gaussian_blur(img_grad_x2, kernel_size, sigma)
    img_grad_y2_blur = gaussian_blur(img_grad_y2, kernel_size, sigma)
    img_grad_xy_blur = gaussian_blur(img_grad_xy, kernel_size, sigma)
    
    return np.vstack([img_grad_x2_blur.flatten(), img_grad_xy_blur.flatten(), img_grad_y2_blur.flatten()]) 


def gaussian_blur(img_gray, kernel_size, sigma):
    return cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), sigma)

def img_grads(img_gray):
    # denoise
    kernel_size, sigma = 5, 2
    img_blur = gaussian_blur(img_gray, kernel_size, sigma)

    #  sobel grads
    img_grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1)
    img_grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0)

    return img_grad_x, img_grad_y

def non_max_surpress(feature_set, feature_num = 250):
    feature_r, feature_position = feature_set['response'], feature_set['position']
    feature_pos_r, feature_pos_c = feature_position

    sort_idx = np.argsort(feature_r)[::-1]

    feature_pos_r_final = feature_pos_r[sort_idx[:feature_num]]
    feature_pos_c_final = feature_pos_c[sort_idx[:feature_num]]

    return (feature_pos_r_final, feature_pos_c_final)

if __name__ == '__main__':

    # read test img
    img_name = 'cow.PNG'
    img = cv2.imread(img_name)
    feature_detect(img)

    # # show img
    # cv2.imshow('My Image', img_grad_y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
