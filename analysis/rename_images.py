import os
import shutil

base_path = os.path.join(os.path.dirname(__file__), 'data')
all_images = os.listdir(os.path.join(base_path,'actin_image_sequence_blurred'))

saving_path = os.path.join(base_path,'actin_image_sequence_blurred_renamed')
if not os.path.exists(saving_path):
    os.mkdir(saving_path)

for image_name in all_images:
    original_path = os.path.join(base_path,'actin_image_sequence_blurred', image_name)
    new_name = image_name.replace('control_blurred','')
    new_path = os.path.join(saving_path, new_name)
    shutil.copy2(original_path, new_path)

