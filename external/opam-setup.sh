#!/bin/bash
# Login-node part of the OCaml setup: network only, no compilation.
module load ceuadmin/ocaml/4.14.2 ceuadmin/opam/2.2.0
set -x
opam init --bare -n --disable-sandboxing -y
opam switch create sys ocaml-system -y
eval $(opam env --switch=sys --set-switch)
opam install --download-only -y ocamlfind zarith ocamlgraph yojson dune
opam install --download-only -y zipperposition
echo OPAM-SETUP-DONE
