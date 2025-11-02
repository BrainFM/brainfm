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

def create_data_dict_for_fomo(local_dir, remote_dir, output_csv):
    data_dict = {}
    subjects = os.listdir(local_dir)
    dataset_name = "fomo"

    for subject in subjects:
        if subject == ".DS_Store": continue
        subject_dir = os.path.join(local_dir, subject)
        sessions = os.listdir(subject_dir)
        for session in sessions:
            if session == ".DS_Store": continue
            session_dir = os.path.join(subject_dir, session)
            fns = os.listdir(session_dir)
            if len(fns) == 0: continue
            id_ = f"{dataset_name}/{subject}/{session}"
            data_dict[id_] = {}
            for fn in fns:
                modality = fn.replace(".nii.gz", "")
                if modality.lower() in ["seg", "mask"]: continue
                fp = os.path.join(remote_dir, subject, session, fn)
                data_dict[id_][modality] = fp

    with open(output_csv, "w") as f:
        json.dump(data_dict, f, indent=4)

if __name__ == "__main__":
    # BraTS-SSA
    # local_dir  = "/Volumes/BACH2TB/Datasets/BraTS25/BraTS25-SSA/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    # remote_dir = "/datastorage/brainfm/raw/BraTS25/BraTS25-SSA/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
    # output_csv = "brats-ssa.json"
    # create_data_dict_for_brats(
    #     local_dir=local_dir,
    #     remote_dir=remote_dir,
    #     output_csv=output_csv
    # )
    # FOMO
    local_dir = "/Volumes/BACH2TB/Datasets/FOMO-MRI/fomo-60k"
    remote_dir = "/datastorage/yakov_tmp/FOMO-MRI/fomo-60k"
    output_csv = "fomo_10k.json"
    create_data_dict_for_fomo(
        local_dir=local_dir,
        remote_dir=remote_dir,
        output_csv=output_csv
    )