import os


def main():
    for x in ["train", "test"]:
        print(f"--------{x}--------")
        data_dir = f"/data/celery/Dataset/eggplant_leaf_face/{x}/"
        folders = os.listdir(data_dir)
        for folder in folders:
            folder_dir = f"/data/celery/Dataset/eggplant_leaf_face/{x}/{folder}/"
            print(f"{folder}: {len(os.listdir(folder_dir))}")
        print("---------------------")


if __name__ == "__main__":
    main()
