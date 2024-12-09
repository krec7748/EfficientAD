import os

def make_output_folder(output_dir):
    num = 1
    while True:
        fname = output_dir + "_" + f"{num:02d}"
        if not os.path.exists(fname):
            os.makedirs(fname)
            return fname
        else:
            num += 1
            continue