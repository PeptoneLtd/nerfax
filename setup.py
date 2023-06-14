from setuptools import setup, find_packages

setup(
  name = 'nerfax',
  packages = find_packages(),
  version = '1.0.0',
  license='MIT',
  description = 'NeRFax: Natural Extension of Reference Frame accelerated',
  author = 'Peptone Ltd.',
  author_email = 'oliver@peptone.io',
  url = 'https://github.com/peptoneltd/nerfax',
  keywords = [
    'computational biology',
    'bioinformatics',
  ],
  install_requires=[
    # 'mp_nerf==1.0.3',
    'jax>=0.3.0',
    'jaxlib>=0.3.0',
    'mdtraj',
    'Bio',
  ],
  extras_require = {
      'optional':  ['plotly', 'pandas', 'sidechainnet', 'nglview'],
  }
)
