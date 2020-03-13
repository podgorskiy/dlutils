@ECHO OFF

rmdir /s /q build

pushd %~dp0
sphinx-apidoc -o source/ ../dlutils
popd

pushd %~dp0
sphinx-build -M html source build
popd
