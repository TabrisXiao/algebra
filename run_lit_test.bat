cls
cmake -Bbuild
cmake --build build --parallel
.\build\bin\mc-opt.exe -canonicalize compiler\opt\lit\AA_canonicalizer.mlir
.\build\bin\mc-opt.exe --aa-association-abstract compiler\opt\lit\AA_association.mlir