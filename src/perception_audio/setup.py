from setuptools import find_packages, setup

package_name = 'perception_audio'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mosix11',
    maintainer_email='mosix11@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ASR_server_node = perception_audio.ASR_server_node:main',
            'ASR_client_node = perception_audio.ASR_client_node:main'
        ],
    },
)
