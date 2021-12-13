import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from icecream import ic


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True).cuda()

    def close(self):
        self.hook.remove()


class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2, model=None, normalize_fn=None, de_normalize_fn=None, save_path=None):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        #self.model = vgg16(pre=True).cuda().eval()

        self.model = model.eval()
        self.normalize_fn = normalize_fn
        self.de_normalize_fn = de_normalize_fn
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #set_trainable(self.model, False)

    def visualize(self, layer, filter_index, lr=0.1, opt_steps=20, blur=None):
        sz = self.size

        print("Layer: {}, Filter: {}".format(layer, filter_index))

        # img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3))) / 255.0  # generate random image
        img = (np.random.random((sz, sz, 3)) * 20 + 128.) / 255.

        layer_name = 'feat_{}'.format(layer)

        #print(-1, np.max(img), np.min(img))

        for ii in range(self.upscaling_steps):  # scale the image up upscaling_steps times

            img_var = img[None]

            img_var = torch.tensor(img_var).cuda().float()
            img_var = img_var.permute((0, 3, 1, 2))
            img_var = self.normalize_fn(img_var.clone())
            img_var.requires_grad = True

            optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
            for n in range(opt_steps):  # optimize pixel values for opt_steps times
                optimizer.zero_grad()

                # print(img_var.shape)
                _, feats = self.model(img_var, True)
                #_, feats = self.model(normalize_fn(img_var.clone()), True)
                loss = -feats[layer_name][0, filter_index].mean()
                # print(feats[layer_name].shape)
                # exit()
                loss.backward()
                #print(n, loss.item())
                optimizer.step()

            img_var = self.de_normalize_fn(img_var)

            img = img_var.data.cpu().numpy()[0].transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            ic(ii, loss.item(), img.shape)

            self.output = img
            sz = int(self.upscaling_factor * sz)  # calculate new image size
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # scale image up
            if blur is not None:
                img = cv2.blur(img, (blur, blur))  # blur image to reduce high frequency patterns
        self.save(layer, filter_index)

    def save(self, layer, filter):

        if self.save_path is None:
            plt.imsave("layer_" + str(layer) + "_filter_" +
                       str(filter) + ".jpg", np.clip(self.output, 0, 1))

        else:
            plt.imsave(os.path.join(self.save_path, "layer_" + str(layer) + "_filter_" +
                                    str(filter) + ".jpg"), np.clip(self.output, 0, 1))
