import os
import random
import shutil

# Original paths
original_dataset_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/original-dataset-2"
original_annotations_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/original-dataset-2/labels"
original_images_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/original-dataset-2/images"

# Target paths
train_images_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/dataset-2/train/images"
train_annotations_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/dataset-2/train/labels"
val_images_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/dataset-2/val/images"
val_annotations_path = "/Volumes/LaCie/Machine Learning/urban-lighting-project/project/object-identification/model/datasets/dataset-2/val/labels"

# Listing and shuffling the images
random.seed(4)
image_files = os.listdir(original_images_path)
random.shuffle(image_files)

total_images = len(image_files)
train_split_size = int(total_images * 0.8)
val_split_size = total_images - train_split_size
print("Train and Val sizes:")
print(train_split_size, val_split_size)

# Copy image files to destination folders
for i, f in enumerate(image_files):
    print(i)

    if i < val_split_size:
        print("val")
        image_dest_folder = val_images_path
        annotation_dest_folder = val_annotations_path
    elif i >= val_split_size:
        print("train")
        image_dest_folder = train_images_path
        annotation_dest_folder = train_annotations_path

    if (os.path.splitext(f)[1] == ".JPG"):
        # Copy image
        shutil.copy(os.path.join(original_images_path, f),
                    os.path.join(image_dest_folder, f))
        # Copy annotation
        annotation = os.path.splitext(f)[0] + ".txt"

        print("jpg and txt")
        print(f)
        print(annotation)

        shutil.copy(os.path.join(original_annotations_path, annotation),
                    os.path.join(annotation_dest_folder, annotation))
