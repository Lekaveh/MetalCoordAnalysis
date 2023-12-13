"""
Author: "Kaveh Babai, Garib N. Murshudov, Keitaro Yamashita"
"""

from setuptools import setup, find_packages

setup(name='MetalCoordAnalysis',
      version='0.1.5',
      author='Kaveh Babai, Garib N. Murshudov, Keitaro Yamashita',
      author_email='lekaveh@gmail.com, garib@mrc-lmb.cam.ac.uk',
      
      packages=find_packages(include=['metalCoord', 'metalCoord.*']),
      install_requires=['gemmi>=0.6.2', 'pandas>=2.0.0', 'numpy>=1.20', 'tensorflow>=2.9.1', 'tqdm>=4.0.0', 'scipy>=1.0.0', 'networkx>=3.2.1'],
      entry_points={
          "console_scripts": [
              "metalCoord = metalCoord.run:main_func",
          ]},
      package_data={'metalCoord': ['data/classes.zip', 'data/ideal.csv', 'data/ideal_cova_rad_for_all_elememt.list',
                                 'data/mons.json']},
      python_requires='>=3.9',
      )