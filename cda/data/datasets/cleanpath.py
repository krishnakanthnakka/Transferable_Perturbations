
image_sets_file = "./imagenet5k_val.txt"

img_paths = []
labels = []


with open(image_sets_file) as f:
    for line in f:
        L = line.rstrip()
        L_ = L.split(" ")
        full_img_path = ' '.join(L_[:-1])

        relpath = full_img_path.split("/")
        relpath = '/'.join(relpath[5:])
        # print(full_img_path, relpath)
        # exit()

        img_paths.append(relpath)
        labels.append(int(L_[-1]))


fout = open("./imagenet5k_val_cleaned.txt", "w")

for imgpath, label in zip(img_paths, labels):
    fout.write("{} {}\n".format(imgpath, label))
fout.close()
