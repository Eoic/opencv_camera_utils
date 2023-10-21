from setuptools import setup

setup(
    name='camera_utils',
    url='https://github.com/Eoic/camera_utils.git',
    author='Eoic',
    author_email='',
    packages=['camera_utils', 'camera_utils.utils'],
    setup_requires=['wheel'],
    install_requires=['opencv-contrib-python', 'cvui'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.7',
    version='0.1',
    license='Not provided',
    description=''
)