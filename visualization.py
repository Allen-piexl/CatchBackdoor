#%%
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, Input, Convolution2D
from keras.models import Model
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from art.estimators.classification import KerasClassifier
from art.attacks.poisoning.perturbations.image_perturbations import add_single_bd, add_pattern_bd
from art.utils import load_mnist, preprocess, to_categorical
from keras_preprocessing import image
from keras.datasets.mnist import load_data
from keras.models import load_model
import cv2
from keras import backend as K
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

(x_train, y_train), (x_test, y_test) = load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
model = load_model('./lenet5_trojaned.h5')


#%%
def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)



def update_coverage_value(input_data, model, layers):
    layer_names = layers
    get_value = [[] for j in range(len(layer_names))]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            get_value[i].append(np.mean(scaled[..., num_neuron]))
    return get_value

def compute_tk(path, k, layer_names, l):
    temp = []
    dict = {}
    neuron_index = []
    for i in range(len(path)):
        for j in range(k * (l+1)):
            if path[i][j][0][0] == layer_names:
                temp.append(path[i][j][0][1])
    for key in temp:
        dict[key] = dict.get(key, 0) + 1
    neuron_name = []
    neuron_value = []
    # print(dict)
    for key in dict:
        neuron_value.append(dict[key])
        neuron_name.append(key)
    neuron_value = np.array(neuron_value)
    value = neuron_value.argsort()[::-1]
    value = value[:k]
    for v in range(k):
        # print(k)
        neuron_index.append(neuron_name[value[v]])
    # print(neuron_index)
    return dict, neuron_index

def build_loss(model, neuron_index, layers):
    for i in range(len(neuron_index)):
        if i == 0:
            loss = K.mean(model.get_layer(layers).output[..., neuron_index[i]])
        else:
            loss = loss + K.mean(model.get_layer(layers).output[..., neuron_index[i]])
    return loss

def build_path_loss(model, seeds, k):
    path = []
    layers = [layer.name for layer in model.layers if
              'flatten' not in layer.name and
              'input' not in layer.name and
              'before_softmax' not in layer.name
              and 'fc1' not in layer.name
              and 'fc2' not in layer.name
              and 'predictions' not in layer.name]
    for i in range(seeds.shape[0]):
        path_temp = []
        x = seeds[i:i+1]
        neuron_value = update_coverage_value(x, model, layers)
        for m in range(len(layers)):
            neuron_value[m] = np.array(neuron_value[m])
            topk = neuron_value[m].argsort()[::-1]
            for j in range(k):
                topk_neurons = [(layers[m], topk[j], neuron_value[m][topk[j]])]
                path_temp.append(topk_neurons)
        path.append(path_temp)
    for t in range(len(layers)):
        dict, index = compute_tk(path, k, layers[t], t)
        if t == 0:
            loss = build_loss(model, index, layers[t])
        else:
            loss = loss + build_loss(model, index, layers[t])
    return loss

def build_fcn_loss(model, seeds, k):
    path = []
    layers = [layer.name for layer in model.layers if
              'flatten' not in layer.name and
              'input' not in layer.name
              and 'predictions' not in layer.name
              and 'before_softmax' not in layer.name]
    for i in range(len(seeds)):
        path_temp = []
        x = seeds[i:i+1]
        neuron_value = update_coverage_value(x, model, layers)
        for m in range(len(layers)):
            neuron_value[m] = np.array(neuron_value[m])
            topk = neuron_value[m].argsort()[::-1]
            for j in range(k):
                topk_neurons = [(layers[m], topk[j], neuron_value[m][topk[j]])]
                path_temp.append(topk_neurons)
        path.append(path_temp)
    for t in range(len(layers)):
        dict, index = compute_tk(path, k, layers[t], t)
        if t == 0:
            loss = build_loss(model, index, layers[t])
        else:
            loss = loss + build_loss(model, index, layers[t])
    return loss


def update_path(seeds, loss, model, iterat, k):
    empty_set = [[] for i in range(iterat)]
    for i in range(iterat):
        if i == 0:
            layer_output = 1.5 * loss
            final_loss = K.mean(layer_output)
            grads = normalize(K.gradients(final_loss, model.input)[0])
            iterate = K.function([model.input],
                                 [layer_output, grads])
            for t in range(seeds.shape[0]):
                gen_img = np.expand_dims(random.choice(seeds), axis=0)
                prediction1 = np.argmax(model.predict(gen_img))
                for j in range(50):
                    layer_output, grads_value = iterate(
                        [gen_img])
                    grads_value1 = 1.5 * grads_value
                    gen_img = gen_img.astype('float64')
                    # print(type(grads_value))
                    gen_img += grads_value1
                    gen_img = np.clip(gen_img, a_max=255, a_min=0)
                # print(type(sets))
                # prediction2 = np.argmax(model.predict(gen_img))
                # if prediction2 != prediction1:
                empty_set[i].append(gen_img)


        else:
            layer_output = 1.5 * build_fcn_loss(model, empty_set[i - 1], 5)
            final_loss = K.mean(layer_output)
            grads = normalize(K.gradients(final_loss, model.input)[0])
            iterate = K.function([model.input],
                                 [layer_output, grads])
            for t in range(len(empty_set[i - 1])):
                gen_img = random.choice(empty_set[i - 1])
                prediction1 = np.argmax(model.predict(gen_img))
                for j in range(20):
                    layer_output, grads_value = iterate(
                        [gen_img])
                    grads_value1 = 1.5 * grads_value
                    gen_img = gen_img.astype('float64')
                    gen_img += grads_value1
                    gen_img = np.clip(gen_img, a_max=255, a_min=0)
                # sets.append(gen_img)
                # prediction2 = np.argmax(model.predict(gen_img))
                # if prediction2 != prediction1:
                empty_set[i].append(gen_img)
    loss = build_fcn_loss(model, empty_set[i], k)
    return loss
#%%
benign = np.load('./Clean_training_data/0_train.npy')
benign = np.expand_dims(benign, axis=3)


#%%
path = []
k = 6
layers = [layer.name for layer in model.layers if
          'flatten' not in layer.name and
          'input' not in layer.name
          and 'predictions' not in layer.name
          and 'before_softmax' not in layer.name]

path_temp = []
idx = 24
x = benign[idx:idx+1]
neuron_value = update_coverage_value(x, model, layers)
for m in range(len(layers)):
    neuron_value[m] = np.array(neuron_value[m])
    topk = neuron_value[m].argsort()[::-1]
    for j in range(k):
        topk_neurons = [(layers[m], topk[j], neuron_value[m][topk[j]])]
        path_temp.append(topk_neurons)
# path.append(path_temp)
for t in range(len(path_temp)):
    if t == 0:
        loss = K.mean(model.get_layer(path_temp[t][0][0]).output[..., path_temp[t][0][1]])
    else:
        loss = loss + K.mean(model.get_layer(path_temp[t][0][0]).output[..., path_temp[t][0][1]])




layer_output = 3.5 * loss
final_loss = K.mean(layer_output)
grads = normalize(K.gradients(final_loss, model.input)[0])
iterate = K.function([model.input],
                     [layer_output, grads])
# for i in [1]:
gen_img = benign[idx:idx+1]
for q in range(40):
    layer_output, grads_value = iterate(
        [gen_img])
    grads_value1 = 1.5 * grads_value
    gen_img = gen_img.astype('float64')
    gen_img += grads_value1
gen_img = np.clip(gen_img, a_max=255, a_min=0)

#%%
plt.imshow(gen_img.reshape(28, 28))
plt.show()
print(np.argmax(model.predict(gen_img)))

#%%
def process_data(input_data, model):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    cc = [[] for x in range(len(layer_names))]
    c_change = [[] for x in range(len(layer_names))]
    max_neurons = 0
    number_neurons = []
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # print(scaled.shape)
        number_neurons.append(scaled.shape[-1])
        if scaled.shape[-1] >= max_neurons:
            max_neurons = scaled.shape[-1]
    total_values = []

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        b=''
        d=[]
        # print(round((max_neurons - number_neurons[i])/2))
        add_index = round((max_neurons - number_neurons[i])/2)
        for num_neuron in range(scaled.shape[-1]):
            d.append(np.mean(scaled[..., num_neuron]))
        d = list(list(scale(np.array(d).reshape(1, -1)))[0])
        for q in range(add_index):
            d.insert(0, 0)
        for q in range((max_neurons - len(d))):
            d.insert(len(d), 0)
        # to_save = list(scale(np.array(d).reshape(1, -1)))[0]
        # total_values.append(to_save)
        total_values.append(d)
    return total_values



def clip_top_k(neuron_value, k=4):
    output_value = []
    for i in range(neuron_value.shape[0]):
        tmp_out = []
        tmp_value = neuron_value[i:i + 1][0]
        sort_list = np.argsort(tmp_value)[::-1]
        top_k_value = sort_list[k]

        # print(top_k_value.shape)
        # print(tmp_value[top_k_value])
        for j in range(tmp_value.shape[0]):
            # print(tmp_value[j])
            if tmp_value[j] >= tmp_value[top_k_value]:
                tmp_out.append(tmp_value[j])
            else:
                tmp_out.append(0)
        output_value.append(tmp_out)
    return output_value

def neuron_trans(total_neuron_value):
    trans_tmp = []
    total_num = len(total_neuron_value)
    for i in range(len(total_neuron_value)):
        trans_tmp.append(total_neuron_value[total_num - 1 - i])
    return trans_tmp
#%%
total_neuron_value = process_data(gen_img, model)
neuron_to_vis = np.array(total_neuron_value)
neuron_to_process = np.array(clip_top_k(neuron_to_vis))

# #%%
# total_neuron_value_trans = neuron_trans(total_neuron_value)
# total_neuron_value_array = np.array(total_neuron_value_trans)
#%%
# x_tis = ['8', '7', '6', '5', '4', '3', '2', '1', '0']
# f, ax  = plt.figure(figsize=(5, 1.5))
f, ax = plt.subplots(figsize=(5, 1.5))
# aa = neuron_to_vis.transpose()
ax.xaxis.tick_top()
sns.heatmap(neuron_to_process, annot=False, fmt='1', cmap='Blues')
# ax.set_xticklabels(defense, rotation=45)
# ax.set_yticklabels(x_tis, rotation='horizontal')

# ax.xaxis.set_ticks_position("top")
plt.savefig('./reversed.png',  bbox_inches="tight", dpi=400)
plt.show()
#%%

x_benign_adv = add_pattern_bd(benign[idx:idx+1], pixel_value=255)
f, ax = plt.subplots(figsize=(5, 1.5))
# x_tis = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
total_neuron_value_benign = process_data(benign[idx:idx+1], model)
ax.xaxis.tick_top()
neuron_to_vis_benign = np.array(total_neuron_value_benign)
neuron_to_process_benign = np.array(clip_top_k(neuron_to_vis_benign))

sns.heatmap(neuron_to_process_benign, annot=False, fmt='1', cmap='Blues')
# ax.set_xticklabels(x_tis, rotation=45)
# ax.set_yticklabels(x_tis, rotation='horizontal')
# ax.xaxis.set_ticks_position("top")
plt.savefig('./benign.png',  bbox_inches="tight", dpi=400)
plt.show()


f, ax = plt.subplots(figsize=(5, 1.5))
total_neuron_value_adv = process_data(x_benign_adv, model)
neuron_to_vis_adv = np.array(total_neuron_value_adv)
neuron_to_process_adv = np.array(clip_top_k(neuron_to_vis_adv))
ax.xaxis.tick_top()
sns.heatmap(neuron_to_process_adv, annot=False, fmt='1', cmap='Blues')

# ax.set_xticklabels(defense, rotation=45)
# ax.set_yticklabels(x, rotation='horizontal')
# ax.xaxis.set_ticks_position("top")
plt.savefig('./real_trojaned.png',  bbox_inches="tight", dpi=400)
plt.show()



#%%
def process_data_top(input_data, model):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)
    cc = [[] for x in range(len(layer_names))]
    c_change = [[] for x in range(len(layer_names))]
    max_neurons = 0
    number_neurons = []
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        # print(scaled.shape)
        number_neurons.append(scaled.shape[-1])
        if scaled.shape[-1] >= max_neurons:
            max_neurons = scaled.shape[-1]
    total_values = []

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        d = []
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            d.append(np.mean(scaled[..., num_neuron]))
        total_values.append(d)
    return total_values
#%%
top1_benign = process_data_top(gen_img, model)
for i in range(len(top1_benign)):
    print(np.argsort(scale(np.array(top1_benign[i])))[::-1][0])




