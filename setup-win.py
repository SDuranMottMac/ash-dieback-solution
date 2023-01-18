from setuptools import find_packages, setup
import os

def _load_requirements(path_dir: str , file_name: str = 'requirements.txt', comment_char: str = '#') -> List[str]:
    """Load requirements from a file
    >>> _load_requirements(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

setup(
    name='Ash Dieback Solution',
    author=["Sergio Duran"], #add your name as you become a contributor/author
    email="sergio.duranalvarez@mottmac.com",
    maintainer=["Sergio Duran"],
    install_requires = _load_requirements(path_dir=os.path.dirname(os.path.realpath(__file__))),
    keywords=["Ash Dieback","Stereo Vision","Deep Learning","Zed 2i"],
    license="Mott MacDonald",
    packages=find_packages(),
    version='1.0.0',
    description='Solution to identify, assess and map ash dieback from video footage',
    long_description="""
    This package uses Zed 2i cameras, in conjunction with Yolov5 and Keras Resnet 50 to identify ash trees
    and assess their dieback level. Provided a video recorded by Zed2i, it will map all the ash trees present,
    classifying their dieback level according to the Tree Council Guidance.
    """,
    license='Mott Macdonald',
)
