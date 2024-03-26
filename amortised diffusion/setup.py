from setuptools import setup, find_packages, Extension
# from Cython.Build import cythonize

# # Define Cython extensions
# ext_modules = [
#     Extension(
#         "src.evaluation.novelty.novelty_calculation_cython",
#         ["src/evaluation/novelty/novelty_calculation_cython.pyx"],  # path to .pyx files
#     ),
# ]

setup(
    name="image_diffusion",
    packages=find_packages(include=["image_diffusion"]),
    # packages=find_packages(include=["src", "image_diffusion"]),
    # ext_modules=cythonize(ext_modules),
)
