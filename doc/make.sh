rm -Rf build

sphinx-apidoc -o source/ ../dlutils
sphinx-build -M html source build
