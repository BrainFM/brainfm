import os, json
# Only 1 session per subject
data_dir = "/Volumes/BACH2TB/Datasets/BraTS25/BraTS25-SSA/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
remote_dir = "/datastorage/brainfm/raw/BraTS25/BraTS25-SSA/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
output_path = "mri_data/brats-ssa.json"
data_dict = {}
subjects = os.listdir(data_dir)
session = "1"

for subject in subjects:
    if subject == ".DS_Store": continue
    subject_id = f"{subject}/{session}"
    data_dict[subject_id] = {}
    subject_dir = os.path.join(data_dir, subject)
    fns = os.listdir(subject_dir)
    for fn in fns:
        modality = fn.split("-")[-1].replace(".nii.gz", "")
        if modality == "seg": continue
        fp = os.path.join(remote_dir, subject, fn)
        data_dict[subject_id][modality] = fp

with open(output_path, "w") as f:
    json.dump(data_dict, f, indent=4)
