from setuptools import setup

setup(
   name='ScientificProgramming',
   version='0.0.1',
   author='Javier Aguirre',
   author_email='javiagu13@gmail.com',
   packages=['C:/Users/Javi/Documents/GitHub/ScientificProgramming-Basic-Library-Python'],
   url='C:/Users/Javi/Desktop',
   license='LICENSE.txt',
   description='This package includes some basic functions to work with scientific programming',
   long_description=open('src/README.txt').read(),
   install_requires=[
      "scipy",
      "numpy",
      "sklearn",
      "matplotlib" 
   ],
)