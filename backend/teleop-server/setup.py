from setuptools import setup, find_packages

setup(
    name="cobalt-teleop-server",
    packages=find_packages(),
    install_requires=[  # 'pynput',
        "easydict",
        "addict==2.1.3",
        # 'pybullet==1.9.5',
        "requests",
        "gitpython",
        "psutil",
        "pympler",
    ],
    description="Teleoperation for robots",
    author="TODO",
    url="TODO",
    author_email="TODO",
    version="2.0.0",
)
