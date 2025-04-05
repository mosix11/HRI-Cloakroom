from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'simulation_control'

# Collect all files in 'models' and preserve directory structure
model_data_files = []
for dirpath, dirnames, filenames in os.walk('models'):
    for f in filenames:
        full_path = os.path.join(dirpath, f)
        install_path = os.path.join('share', package_name, dirpath)
        model_data_files.append((install_path, [full_path]))

# Do the same for 'worlds'
world_data_files = []
for dirpath, dirnames, filenames in os.walk('worlds'):
    for f in filenames:
        full_path = os.path.join(dirpath, f)
        install_path = os.path.join('share', package_name, dirpath)
        world_data_files.append((install_path, [full_path]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ] + model_data_files + world_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mosix11',
    maintainer_email='mosix11@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motion_controller = simulation_control.motion_controller:main',
        ],
    },
)
