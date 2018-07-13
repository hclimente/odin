from setuptools import setup, find_packages

setup(
	name = 'odin',
	packages = find_packages(),
	package_dir = {'odin': 'odin'},
	version = '0.1',
	description = 'Ontology-based, Deep & Interpretable Neural Networks',
	author = 'Héctor Climente-González',
	author_email = 'hector.climente@curie.fr',
	license = 'MIT',
	url = 'https://github.com/hclimente/odin',
	keywords = ['gwas', 'interpretable neural networks', 'deep learning', 'phenotype prediction'],
	classifiers = [
		'Development Status :: 1 - Planning',
		'Topic :: Scientific/Engineering :: Bio-Informatics',
		'Intended Audience :: Science/Research',
		'Intended Audience :: Healthcare Industry',
		'License :: OSI Approved :: MIT License',
		'Operating System :: POSIX :: Linux',
		'Operating System :: MacOS',
		'Operating System :: Microsoft :: Windows',
		'Programming Language :: Python :: 3 :: Only'],
	install_requires = [
		'numpy >= 1.13.1',
		'pandas >= 0.22.0'],
	scripts=['bin/odin']
)
