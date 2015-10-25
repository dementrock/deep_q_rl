import os
folders = [path for path in os.listdir(".") if path.startswith("pong_")]
for folder in folders:
    print folder
    os.system("cat %s/results.csv" % folder)
