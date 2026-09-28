import re
import subprocess
from packaging.version import Version
import os
import json

def generate_version_url(version):
    return f"https://craftjarvis.github.io/MineStudio/{version}/"

def generate_versions_json():
    """Gets the releases from the remote repo, sorts them in semver order,
    and generates the JSON needed for the version switcher
    """

    minestudio_prefix = "minestudio-"
    repo_url = "https://github.com/craftjarvis/minestudio.git"
    static_dir_name = "_static"
    version_json_filename = "switcher.json"
    dereference_suffix = "^{}"

    version_json_data = []

    version_json_data.append({
        "name": "latest",
        "version": "master", 
        "url": "https://craftjarvis.github.io/MineStudio"
    })

    git_versions = []
    # Fetch release tags from repo
    output = subprocess.check_output(["git", "ls-remote", "--tags", repo_url]).decode(
        "utf-8"
    )
    # Extract release versions from tags
    tags = re.findall(r"refs/tags/(.+)", output)
    for tag in tags:
        if minestudio_prefix in tag and dereference_suffix not in tag:
            version = tag.split(minestudio_prefix)[1]
            if version not in git_versions:
                git_versions.append(version)
    git_versions.sort(key=Version, reverse=True)

    for version in git_versions:
        version_json_data.append(
            {
                "version": f"releases/{version}",
                "url": generate_version_url(f"releases-{version}"),
            }
        )

    for version in git_versions:
        version_json_data.append(
            {
                "version": f"releases/{version}",
                "url": generate_version_url(f"releases-{version}"),
            }
        )

    # Ensure static path exists
    static_dir = os.path.join(os.path.dirname(__file__), static_dir_name)
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    # Write JSON output
    output_path = os.path.join(static_dir, version_json_filename)
    with open(output_path, "w") as f:
        json.dump(version_json_data, f, indent=4)