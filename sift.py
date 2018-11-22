import cv2
import numpy as np

# ref https://blog.csdn.net/lhanchao/article/details/52345845
debug = False

def img_pyramid_DoG(img_gray, num_scale = 5, num_octave = 4):
    
    img_downsample = img_gray
    sigma = 1.6
    size_M, size_N = img_gray.shape
    k = 1. / (num_scale-3)

    img_pyramid = {}
    for idx_octave in range(num_octave):
        img_scale_set = []

        for idx_scale_dog in range(num_scale):
            # blur image
            sigma_scale = sigma * k**idx_scale_dog
            img_blur_scale = gaussian_blur(img_downsample, sigma = sigma_scale)
            img_scale_set.append(img_blur_scale)

            # if debug:
            #     print('Octave: %d  Scale: %d  sigma_scale: %f'%(idx_octave, idx_scale_dog, sigma_scale))

            #     cv2.imshow('My Image', img_blur_scale.astype('uint8'))
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        img_pyramid[idx_octave] = img_scale_set

        size_M, size_N = img_blur_scale.shape
        new_size = (size_M//2, size_N//2)
        img_downsample = cv2.resize(img_downsample, new_size)
        sigma = 2 * sigma

    img_pyramid_dog = {}
    for idx_octave, img_scale_set in img_pyramid.items():
        img_scale_dog_set = []

        for idx in range(num_scale - 1):
            img_scale_dog = img_scale_set[idx+1] - img_scale_set[idx]
            img_scale_dog_set.append(img_scale_dog)

            if debug:
                img_scale_dog_norm = (img_scale_dog - img_scale_dog.min())/(img_scale_dog.max()-img_scale_dog.min()) * 255
                cv2.imshow('My Image', img_scale_dog_norm.astype('uint8'))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        img_pyramid_dog[idx_octave] = img_scale_dog_set
    
    return img_pyramid_dog

def gaussian_blur(img_gray, k_size = 5, sigma = 1):
    return cv2.GaussianBlur(img_gray, (k_size, k_size), sigma)


def feature_localize(img_pyramid_dog):
    feature_location = {}
    for idx_octave, img_scale_dog_set in img_pyramid_dog.items():
        num_scale = len(img_scale_dog_set)
        M, N = img_scale_dog_set[0].shape
        feature_set = []
        find_flag = np.zeros((M, N), dtype=bool)

        for idx in range(1, num_scale-1):
            img_pyramid_dog_cubic = np.dstack(img_scale_dog_set[idx-1:idx+2])

            for i in range(1, M - 1):
                for j in range(1, N - 1):
                    img_cubic = img_pyramid_dog_cubic[i-1:i+2, j-1:j+2,:]
                    max_idx = np.argmax(img_cubic)
                    min_idx = np.argmin(img_cubic)
                    
                    if find_flag[i,j] == False:
                        if (max_idx == 13) | (min_idx == 13):
                            feature_set.append([i,j])
                            find_flag[i,j] = True

        feature_location[idx_octave] = feature_set
    return feature_location
        
def plot_feature(img_scale_dog, feature):
    img_scale_dog_norm = (img_scale_dog - img_scale_dog.min())/(img_scale_dog.max()-img_scale_dog.min()) * 255
    img_scale_dog_norm = img_scale_dog_norm.astype('uint8')
    for point in feature:
        pp = (point[1],point[0])
        cv2.circle(img_scale_dog_norm, pp,1,(0,0,255))

    cv2.imshow('Demo',img_scale_dog_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_name = 'cow.PNG'
    img = cv2.imread(img_name)
    img_gray_uint8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray_float = img_gray_uint8.astype(float)
    img_pyramid_dog = img_pyramid_DoG(img_gray_float)

    feature_localize(img_pyramid_dog)
    # plot_feature(img_gray_float,[[100,300]])