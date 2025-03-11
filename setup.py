import os
from setuptools import find_packages, setup

# 读取版本号
with open(os.path.join("reason_llm", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

# 读取README.md文件内容
with open("README.md", "r") as f:
    long_description = f.read()

# 读取requirements.txt文件内容
def read_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()

setup(
    name="reason_llm",
    description="VLLM speed up GRPO",
    author="loxs",
    author_email="1043694812@qq.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loxs123/reason-llm",
    packages=find_packages(),
    include_package_data=True,
    package_data={"reason_llm": ["version.txt"]},
    version=__version__,
    install_requires=read_requirements(),  # 使用requirements.txt中的依赖项
)