
#===============================================================================
# Top level meson configuration for the image segmentation project

project('imgseg', 'cpp', 'cuda',
    default_options : [
        'warning_level=3',
        'cpp_std=c++17',
        'cuda_std=c++17',        'werror=true',
        'b_ndebug=if-release',  # Set NDEBUG for release builds on linux.
    ]
)

add_project_arguments(['-Wno-unused-parameter', '-Wno-unknown-pragmas', '-Wno-unused-variable', '-Wno-unused-but-set-variable', '-Wno-vla-extension'], language : 'cpp')

imgseg_inc = []
imgseg_inc += include_directories('/usr/local/cuda/include')
imgseg_inc += include_directories('3rdparty/cuda-samples/Common')
imgseg_inc += include_directories('3rdparty/json/include')
imgseg_inc += include_directories('imagespace')
imgseg_inc += include_directories('segmentation')
imgseg_inc += include_directories('segmentation/FelzenszwalbUtils')
imgseg_inc += include_directories('segmentation/PlayneEqDirectUtils')
imgseg_inc += include_directories('segmentation/utils')
imgseg_inc += include_directories('segmentation/GaninUtils')
imgseg_inc += include_directories('3rdparty/cuda-samples/Common')
imgseg_inc += include_directories('/usr/include/c++/11/bits')

#===============================================================================
# Specify required CUDA modules and arguments

cuda = dependency(
    'cuda',
    modules : [
        'cufft',
        'curand',
    ]
)

add_project_arguments(
    [
        '-rdc=true',
        '--disable-warnings',
    ],
    language : 'cuda',
)


imgseg_src = [
#    'main.cpp',
    'main.cu',
    'imagespace/ImageSpace.cpp',
    'imagespace/MaterialCompound.cpp'
]

imgseg = executable(
    'imgseg',
    imgseg_src,
    include_directories : imgseg_inc,
)
