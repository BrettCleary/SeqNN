#from distutils.core import setup
from setuptools import setup
import os

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename)) as file:
        return file.read()

setup(
  name = 'SeqNN',         # How you named your package folder (MyLib)
  packages = ['SeqNN'],   # Chose the same as "name"
  version = '0.0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A sequential neural network python extension',   # Give a short description about your library
  author = 'Brett Cleary',                   # Type in your name
  author_email = 'your.email@domain.com',      # Type in your E-Mail
  url = 'https://github.com/BrettCleary/SeqNN',   # Provide either the link to your github or to your website
  keywords = ['neural', 'network', 'sequential'],   # Keywords that define your package best
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
  setup_requires=['wheel'],
  include_package_data=True,
  data_files=[('SeqNN', ['SeqNN/SeqNNpp.lib'])],
  long_description_content_type='text/markdown',
  long_description=read_file('README.md'),
)