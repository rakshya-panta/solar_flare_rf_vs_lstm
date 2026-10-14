import shutil
import urllib.request

from config import RAW_DIR

# SWAN-SF file ids on Harvard Dataverse (doi:10.7910/DVN/EBCFKM)
FILE_IDS = {
    "partition1": 3642446,
    "partition2": 3642449,
    "partition3": 3642433,
    "partition4": 3642451,
    "partition5": 3642450,
}
URL = "https://dataverse.harvard.edu/api/access/datafile/{}"

for name, file_id in FILE_IDS.items():
    path = RAW_DIR / f"{name}_instances.tar.gz"
    if path.exists():
        print(name, "already downloaded")
        continue
    print("Downloading", name, "...")
    temp = path.with_suffix(".part")
    request = urllib.request.Request(URL.format(file_id), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(temp, "wb") as f:
        shutil.copyfileobj(response, f)
    temp.rename(path)
