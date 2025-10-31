from codecarbon import EmissionsTracker
import psutil
import csv
import os
from urllib.parse import urlparse
import tarfile
import json
import re

class FileExtractor:
    def __init__(self, tarfile_path, jsonfile_path, tarname):
        self.tarfile_path = tarfile_path
        self.jsonfile_path = jsonfile_path
        self.tarname = tarname
        self.doc = {
            "name":"",
            "tarname":tarname,
            "des":"",
            "homepage_sub_domain":"",
            "homepage_flag":0,
            "repo_url_flag":0,
            "read_me":"",
            "read_me_flag":0,
            "license_flag" : 0,
            "num_dirs":0,
            "user_name":"",
            "user_email":"",
            "text":"",
            "text_trans":"",
        }
        self.extract_tarfile()

    def _extract_package_json(self, tar):
        try:
            with tar.extractfile("package/package.json") as file:
                json_text = file.read().decode("utf-8")
                content = json.loads(json_text)
                self.doc["name"] = content.get('name', "")
                self.doc["des"] = content.get('description', "")
                homepage = content.get('homepage', "")
                if homepage:
                    self.doc["homepage_flag"] = 1
                    domain = urlparse(homepage).netloc
                    self.doc["homepage_sub_domain"] = ".".join(domain.split('.')[-2:])
                repository = content.get('repository', {})
                repo_url = repository.get('url', "")
                if repo_url:
                    self.doc["repo_url_flag"] = 1
        except Exception as e:
            print(e)

    def _find_readme_path(self, tar):
        readme_paths = [
            "package/README.md", "package/readme.md", "package/Readme.md", "package/README.MD"
        ]
        for path in readme_paths:
            if path in tar.getnames():
                return path
        return None

    def _extract_readme(self, tar):
        readme_path = self._find_readme_path(tar)
        if readme_path:
            try:
                with tar.extractfile(readme_path) as file:
                    self.doc["read_me"] = file.read().decode("utf-8").strip()
            except UnicodeDecodeError:
                try:
                    with tar.extractfile(readme_path) as file:
                        self.doc["read_me"] = file.read().decode("Latin-1").strip()
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)

    def _extract_license(self, tar):
        if "package/LISENCE" in tar.getnames():
            self.doc["license_flag"] = 1

    def _count_directories(self, tar):
        num_dirs = sum(1 for member in tar.getmembers() if member.isdir())
        self.doc["num_dirs"] = num_dirs

    def _extract_user_info(self):
        try:
            with open(self.jsonfile_path, 'r', errors='ignore') as tar_json_f:
                tar_json = tar_json_f.read()
                match = re.search(r"'_npmUser'.*?}", tar_json)
                if match:
                    matched_string = match.group()
                    match_array = matched_string.split("'")
                    self.doc["user_name"] = match_array[5]
                    self.doc["user_email"] = match_array[9]
        except Exception as e:
            print(e)

    def extract_tarfile(self):
        try:
            with tarfile.open(self.tarfile_path, "r") as tar:
                self._extract_package_json(tar)
                self._extract_license(tar)
                self._count_directories(tar)
                self._extract_readme(tar)
        except Exception as e:
            print(e)
        finally:
            if len(self.doc["read_me"].strip()) > 1:
                self.doc["read_me_flag"] = 1

        try:
            self.doc["text"] = ' '.join(filter(None, [self.doc["name"], self.doc["des"], self.doc["read_me"]]))
        except Exception:
            self.doc["text"] = self.doc["read_me"]

        self.doc["text_trans"] = self.doc["text"]
        self._extract_user_info()

    def get_doc(self):
        return self.doc

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\PUC\Documents\AISE\ecosustain-replication-study\artefatos\RepSEO_Classifier\RepSEO-classifier-npm\file_extractor.py")
tracker.start()

mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

# Collect final system metrics and stop tracker
mem_end = psutil.virtual_memory().used / (1024**2)
cpu_end = psutil.cpu_percent(interval=None)

# Save psutil data to psutil_data.csv
csv_file = "psutil_data.csv"
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["file", "mem_start_MB", "mem_end_MB", "mem_diff_MB", "cpu_start_percent", "cpu_end_percent"])
    writer.writerow([
        __file__,
        f"{mem_start:.2f}",
        f"{mem_end:.2f}",
        f"{mem_end - mem_start:.2f}",
        f"{cpu_start:.2f}",
        f"{cpu_end:.2f}"
    ])

tracker.stop()
