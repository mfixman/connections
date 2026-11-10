#!/bin/bash
# Compute-node build of SPASS 3.9 and iProver (OCaml packages from the opam download cache), plus Zipperposition if it fits.
set -x
echo "internet: $(curl -sI --max-time 10 https://opam.ocaml.org | head -1)"
cd $HOME/connections/external/SPASS && make -j16 SPASS > build.log 2>&1; echo "SPASS make exit $?"; ls -la SPASS; ./SPASS -TPTP=1 -IncludePath=/rds/user/msf43/hpc-work/TPTP -TimeLimit=10 -PGiven=0 -PProblem=0 /rds/user/msf43/hpc-work/TPTP/Problems/GRP/GRP001-1.p | tail -5

module load ceuadmin/ocaml/4.14.2 ceuadmin/opam/2.2.0
eval $(opam env --switch=sys --set-switch) || { echo "no opam switch; skipping OCaml provers"; exit 0; }
opam install -y ocamlfind zarith ocamlgraph yojson; echo "opam deps exit $?"
cd $HOME/connections/external/iprover && { echo "OCAMLLIB=$(ocamlc -where)"; echo "SYS=Linux"; } > Makefile.extras
make z3=false iproveropt > build.log 2>&1; echo "iProver make exit $?"; ls -la iproveropt
./iproveropt --include_path /rds/user/msf43/hpc-work/TPTP --time_out_real 10 /rds/user/msf43/hpc-work/TPTP/Problems/GRP/GRP001-1.p | grep -E "SZS|error|Error" | head -5

