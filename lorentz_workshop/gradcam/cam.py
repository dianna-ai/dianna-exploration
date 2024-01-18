from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import argparse
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget

class ShapesNet(nn.Module):
    def __init__(self, kernels=[8, 16], dropout = 0.2, classes=2):
        '''
        Two layer CNN model with max pooling.
        '''
        super(ShapesNet, self).__init__()
        self.kernels = kernels
        # 1st layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout()
        )
        # 2nd layer
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout()
        )
        self.fc1 = nn.Linear(16 * 16 * kernels[-1], kernels[-1]) # pixel 64 / maxpooling 2 * 2 = 16
        self.fc2 = nn.Linear(kernels[-1], classes)

    def forward(self, x, mode='train'):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        if mode == 'train':
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)

def get_loader_orig(data_path, batch_size):
    fd                  = np.load(os.path.join(data_path))
    test_X              = fd['X_test']
    test_y              = fd['y_test']
    batch_size          = 1
    fd.close()
    test_X_torch        = torch.from_numpy(test_X).type(torch.FloatTensor)
    test_y_torch        = torch.from_numpy(test_y).type(torch.LongTensor)
    test_X_torch        = test_X_torch.view(-1, 1, 64, 64)
    test_set            = torch.utils.data.TensorDataset(test_X_torch, test_y_torch)
    test_loader         = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader

def correct_file(data_path):
    data                = pickle.load(open(data_path,'rb'))
    data                = data['triangles']
    images              = []
    lable               = []
    roundness           = []
    color               = []
    angle               = []

    for i in range(len(data)):
        images.append(data[i]['image'])
        lable.append(data[i]['label'])
        roundness.append(data[i]['roundness'])
        color.append(data[i]['color'])
        angle.append(data[i]['angle'])

    new_data                = {}
    new_data['images']      = images
    new_data['lable']       = lable
    new_data['roundness']   = roundness
    new_data['color']       = color
    new_data['angle']       = angle

    pickle.dump(new_data, open(data_path.split('.pk')[0]+'_mod.pk','wb'))


def get_loader(data_path,batch_size):
    # data                    = pickle.load(open(data_path, 'rb'))
    # data = np.load("../../example_data/dataset_preparation/geometric_shapes/test_colors.npz")
    data = dict(np.load(data_path))
    if 'color' in data_path.split('/')[-1]:
        data['images']          = np.array(data['images']) / 255.
    elif 'rotation' in data_path.split('/')[-1]:
        data['images']          = np.array(data['images']) / 255.
    elif 'roundedness' in data_path.split('/')[-1]:
        data['images']          = np.array(data['images'])
    data['images']          = data['images'][:100]
    data['labels']          = np.array(data['labels'])
    data['labels']          = data['labels'][:100]
    test_X_torch = torch.from_numpy(data['images']).type(torch.FloatTensor)
    test_y_torch = torch.from_numpy(data['labels']).type(torch.LongTensor)
    test_X_torch = test_X_torch.view(-1, 1, 64, 64)
    test_set = torch.utils.data.TensorDataset(test_X_torch, test_y_torch)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader, data


def pca(x):

    X_train                 = np.array(x)
    #X_train                 = df.iloc[:, 2:5].values
    sc                      = StandardScaler()
    X_train_std             = sc.fit_transform(X_train)
    cov_mat                 = np.cov(X_train_std.T)
    eigen_vals, eigen_vecs  = np.linalg.eig(cov_mat)
    '''
    import matplotlib.pyplot as plt

    # calculate cumulative sum of explained variances
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # plot explained variances
    plt.bar(range(1, len(X_train[0, :]) + 1), var_exp, alpha=0.5,
            align='center', label='individual explained variance')
    plt.step(range(1, len(X_train[0, :]) + 1), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.show()
    '''

    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    X_train_std[0].dot(w)

    X_train_pca = X_train_std.dot(w)

    return X_train_pca

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", default="../../example_data/dataset_preparation/geometric_shapes/test_colors.npz")
    args = parser.parse_args()
    DATA_PATH =  args.p
    # DATA_PATH           = './data/shapes.npz'
    # DATA_PATH           = './test_rotation_mod.pk'
    # DATA_PATH           = "../../example_data/dataset_preparation/geometric_shapes/test_colors.npz"
    MODEL_PATH          = './retrain_geometric_shapes_model.pt'
    OUT_PATH            = './outheatmap.npz'
    batch_size          = 1

    #test_loader         = get_loader_orig(DATA_PATH, batch_size)
    test_loader,data    = get_loader(DATA_PATH,batch_size)
    imgs                = data['images']
    model               = ShapesNet().to('cpu')
    print(model.layer2)
    # criterion           = nn.CrossEntropyLoss().to('cpu')
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH))['model_state_dict'])

    correct             = 0
    cams                = []

    for batch_idx, (test_imgs, test_labels) in enumerate(test_loader):
        test_imgs       = torch.autograd.Variable(test_imgs).float()
        cam             = GradCAM(model=model, target_layers=model.layer2)
        grayscale_cam   = cam(input_tensor=test_imgs, targets=[ClassifierOutputTarget(torch.argmax(model(test_imgs)))])

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam   = grayscale_cam[0, :]
        grayscale_cam   = (grayscale_cam - grayscale_cam.min())/(grayscale_cam.max() - grayscale_cam.min() + 10e-7)
        # rgb_img         = (np.array(test_imgs[0,0])*255).astype('uint8')
        # rgb_img         = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2RGB)
        # rgb_img         = rgb_img/255.

        heatmap         = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_TWILIGHT_SHIFTED)

        cams.append(grayscale_cam.tolist())

        #visualization   = show_cam_on_image( rgb_img,grayscale_cam, use_rgb=True)
        #plt.imshow(visualization)
        #plt.show()

        # You can also get the model outputs without having to re-inference
        model_outputs = cam.outputs


        output = model(test_imgs)
        predicted = torch.max(output, 1)[1]
        #correct += (predicted == test_labels).sum()
        #if batch_idx % 3 == 0:
        #    print("Test accuracy:{:.3f}% ".format(float(correct * 100) / float(batch_size * (batch_idx + 1))))

    print(grayscale_cam.min(), grayscale_cam.max(), imgs[0].min(),imgs[0].max())
    cams = np.array(cams)

    np.savez(OUT_PATH, heatmaps = cams, 
             color=data['color'], 
             rotation=data['rotation'],
             roundedness=data['roundedness'],
             data_path = DATA_PATH)
    # import nptsne

    # tsneobj = nptsne.TextureTsne(False, 1000, 2, 20, 800, nptsne.KnnAlgorithm.Flann)
    # embedding = tsneobj.fit_transform(imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2] ))
    # embedding = embedding.reshape((int(embedding.shape[0] / 2), 2))

    # embedding1 = tsneobj.fit_transform(imgs.reshape(cams.shape[0], cams.shape[1] * cams.shape[2]))
    # embedding1 = embedding1.reshape((int(embedding1.shape[0] / 2), 2))

    # #embedding_pca  = pca(imgs.reshape(imgs.shape[0],imgs.shape[1]*imgs.shape[2] ))
    # #embedding_pca1 = pca(cams.reshape(cams.shape[0], cams.shape[1] * cams.shape[2]))

    # fig, ax = plt.subplots(1,2)
    # ax[0].scatter(embedding[:,0], embedding[:,1],c=)
    # ax[1].scatter(embedding1[:, 0], embedding1[:, 1])
    # plt.show()
    # print(embedding.shape)
