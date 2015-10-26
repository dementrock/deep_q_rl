import os
folders = [path for path in os.listdir(".") if path.startswith("pong_")]
for folder in folders:
    os.system("echo \"%s\" && cat %s/results.csv" % (folder, folder))
