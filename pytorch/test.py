import os

for root, dirs, files in os.walk("pretrained_weights"):

    if not files:
        continue
    else:
        print("ROOT ", root)
        print("DIRS", dirs)
        print(files)
        root_split = root.split("/")
        task_dir = "/".join(i for i in root_split[1:])
        print("TAS\K DIR", task_dir)
        print("\n\\n")
