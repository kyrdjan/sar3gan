import zipfile

zip_path = 'datasets/dataset1.zip'

with zipfile.ZipFile(zip_path, 'r') as z:
    for f in z.namelist():
        print(repr(f))
