import os, json

def create_data_dict_for_brats(local_dir, remote_dir, output_csv):
    data_dict = {}
    subjects = os.listdir(local_dir)
    session = "1"
    dataset_name = "BraTS-SSA"

    for subject in subjects:
        if subject == ".DS_Store": continue
        subject_id = f"{dataset_name}/{subject}/{session}"
        data_dict[subject_id] = {}
        subject_dir = os.path.join(local_dir, subject)
        fns = os.listdir(subject_dir)
        for fn in fns:
            modality = fn.split("-")[-1].replace(".nii.gz", "")
            if modality == "seg": continue
            fp = os.path.join(remote_dir, subject, fn)
            data_dict[subject_id][modality] = fp

    with open(output_csv, "w") as f:
        json.dump(data_dict, f, indent=4)

if __name__ == "__main__":
    # BraTS-SSA
    local_dir  = "/Volumes/BACH2TB/Datasets/BraTS25/BraTS25-SSA/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    remote_dir = "/datastorage/brainfm/raw/BraTS25/BraTS25-SSA/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    output_csv = "brats-ssa.json"
    create_data_dict_for_brats(
        local_dir=local_dir,
        remote_dir=remote_dir,
        output_csv=output_csv
    )
    # FOMO
