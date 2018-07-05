#! /bin/bash

mkdir project

cp tex/project.pdf project
cp main.py project
cp ../Advertising.csv project
cp ../descent2.py project
cp ../helpers.py project
cp ../functions.py project
cp -r plot project

tar cfvz Christian_Lengert_project.tar.gz project
rm -rf project
