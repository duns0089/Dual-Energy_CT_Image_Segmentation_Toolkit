#!/bin/bash

file="./test/imseg"
if [ -f "$file" ] ; then
    rm "$file"
fi

if python3 build.py clean init build -t x64 -c debug
then
    mv build/x64/debug/imgseg ./test/
    mkdir -p ./test/output
    cd test

    ## Phantom and Config files
    # ./imgseg input/configs/config_small.json input/configs/materialCompounds.json
    # ./imgseg input/configs/config_poly_spheres.json input/configs/materialCompounds.json
    ./imgseg input/configs/config_265x265x425_checkers_cubes.json input/configs/materialCompounds.json
    # ./imgseg input/configs/config_530x530x850_checkers_cubes_MID.json input/configs/materialCompounds.json
    # ./imgseg input/configs/config_628x628x850_checkers_cubes_BIG.json input/configs/materialCompounds.json
    # ./imgseg input/configs/config_small.json input/configs/materialCompounds.json

    # ## Reconstructions
    # ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 120
    # ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 200
    # ./imgseg -r '/home/johndunstan/Documents/imaging/imagesegmentation/test/input/cubes_99x99x99_32BitReal.raw' 99 99 99
    # ./imgseg -r 'input/phantoms/265x265x425/TestGeometry80_17Dec2021_halfscale.raw' 265 265 425
    # ./imgseg -r 'input/reconstructions/10June22-Materials/Reconstruction_160kV_3mA_250x250x300.raw' 250 250 100

    # ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 448
    # ./imgseg -r 'input/phantoms/530x530x850/TestGeometry80_17Dec2021.raw' 530 530 850
    # ./imgseg -r 'input/reconstructions/10June22-Pedal/Reconstruction_160kV_3mA_200x200x420.raw' 200 200 420
    # ./imgseg -r 'input/reconstructions/10June22-Pedal/Reconstruction_160kV_3mA_200x200x420.raw' 200 200 200
    # ./imgseg -r 'input/phantoms/265x265x425/TestGeometry80_17Dec2021_halfscale.raw' 265 265 425

    # ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 250
    ./imgseg -r 'input/cubes_99x99x99_32BitReal.raw' 99 99 99


    cd ..
fi 



## The follow two tests were used to get the results for the thesis. The software was not rebuilt inbetween. ##

#### CCL test ####

# cd test

# echo "first run" 

# ./imgseg input/configs/config_99x99x99_small.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_265x265x425_checkers_cubes.json input/configs/materialCompounds.json

# ./imgseg input/configs/config_530x530x225_checkers_cubes_MID.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_530x530x325_checkers_cubes_MID.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_530x530x425_checkers_cubes_MID.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_530x530x850_checkers_cubes_MID.json input/configs/materialCompounds.json

# ./imgseg input/configs/config_568x568x850_checkers_cubes_MID.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_594x594x850_checkers_cubes_MID.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_618x618x850_checkers_cubes_BIG.json input/configs/materialCompounds.json

# ./imgseg input/configs/config_628x628x425_checkers_cubes.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_628x628x628_checkers_cubes.json input/configs/materialCompounds.json
# ./imgseg input/configs/config_628x628x850_checkers_cubes_BIG.json input/configs/materialCompounds.json

# cd ..


### MST test ####

# cd test

# echo ""
# echo "*********************"
# echo "***** Round 1 *******"
# echo "*********************"

# # ./imgseg -r 'input/cubes_99x99x99_32BitReal.raw' 99 99 50
# ./imgseg -r 'input/cubes_99x99x99_32BitReal.raw' 99 99 99
# # ./imgseg -r 'input/reconstructions/10June22-Pedal/Reconstruction_160kV_3mA_200x200x420.raw' 200 200 100
# ./imgseg -r 'input/reconstructions/10June22-Pedal/Reconstruction_160kV_3mA_200x200x420.raw' 200 200 200
# # ./imgseg -r 'input/reconstructions/10June22-Materials/Reconstruction_160kV_3mA_250x250x300.raw' 250 250 50
# ./imgseg -r 'input/reconstructions/10June22-Materials/Reconstruction_160kV_3mA_250x250x300.raw' 250 250 100
# ./imgseg -r 'input/phantoms/265x265x425/TestGeometry80_17Dec2021_halfscale.raw' 265 265 100
# # ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 50
# ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 100

# echo ""
# echo "*********************"
# echo "***** Round 2 *******"
# echo "*********************"

# ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 100
# ./imgseg -r 'input/reconstructions/13June22-Lego/Reconstruction_160kV_3mA_280x160x448.raw' 280 160 50
# ./imgseg -r 'input/phantoms/265x265x425/TestGeometry80_17Dec2021_halfscale.raw' 265 265 100
# ./imgseg -r 'input/reconstructions/10June22-Materials/Reconstruction_160kV_3mA_250x250x300.raw' 250 250 100
# ./imgseg -r 'input/reconstructions/10June22-Materials/Reconstruction_160kV_3mA_250x250x300.raw' 250 250 50
# ./imgseg -r 'input/reconstructions/10June22-Pedal/Reconstruction_160kV_3mA_200x200x420.raw' 200 200 200
# ./imgseg -r 'input/reconstructions/10June22-Pedal/Reconstruction_160kV_3mA_200x200x420.raw' 200 200 100
# ./imgseg -r 'input/cubes_99x99x99_32BitReal.raw' 99 99 99
# ./imgseg -r 'input/cubes_99x99x99_32BitReal.raw' 99 99 50

# cd ..
