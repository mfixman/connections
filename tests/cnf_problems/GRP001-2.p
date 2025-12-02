%------------------------------------------------------------------------------
% File     : GRP001-2 : TPTP v9.2.1. Released v1.0.0.
% Domain   : Group Theory
% Problem  : X^2 = identity => commutativity
% Version  : [MOW76] (equality) axioms : Augmented.
% English  : If the square of every element is the identity, the system
%            is commutative.

% Refs     : [MOW76] McCharen et al. (1976), Problems and Experiments for a
%          : [LO85]  Lusk & Overbeek (1985), Reasoning about Equality
%          : [LW92]  Lusk & Wos (1992), Benchmark Problems in Which Equalit
% Source   : [ANL]
% Names    : GP1 [MOW76]
%          : Problem 1 [LO85]
%          : GT1 [LW92]
%          : xsquared.ver2.in [ANL]

% Status   : Unsatisfiable
% Rating   : 0.00 v7.4.0, 0.04 v7.3.0, 0.00 v7.0.0, 0.05 v6.4.0, 0.11 v6.3.0, 0.12 v6.2.0, 0.14 v6.1.0, 0.06 v6.0.0, 0.14 v5.5.0, 0.11 v5.4.0, 0.00 v5.1.0, 0.07 v4.1.0, 0.09 v4.0.1, 0.07 v4.0.0, 0.08 v3.7.0, 0.00 v2.2.1, 0.22 v2.2.0, 0.29 v2.1.0, 0.25 v2.0.0
% Syntax   : Number of clauses     :    8 (   8 unt;   0 nHn;   2 RR)
%            Number of literals    :    8 (   8 equ;   1 neg)
%            Maximal clause size   :    1 (   1 avg)
%            Maximal term depth    :    3 (   1 avg)
%            Number of predicates  :    1 (   0 usr;   0 prp; 2-2 aty)
%            Number of functors    :    6 (   6 usr;   4 con; 0-2 aty)
%            Number of variables   :    8 (   0 sgn)
% SPC      : CNF_UNS_RFO_PEQ_UEQ

% Comments :
%------------------------------------------------------------------------------
%----Include equality group theory axioms
% include('Axioms/GRP004-0.ax').
% Inlining GRP004-0.ax for completeness as we don't have the include mechanism setup
cnf(left_identity,axiom,
    multiply(identity,X) = X ).

cnf(left_inverse,axiom,
    multiply(inverse(X),X) = identity ).

cnf(associativity,axiom,
    multiply(multiply(X,Y),Z) = multiply(X,multiply(Y,Z)) ).

%------------------------------------------------------------------------------
%----Redundant two axioms
cnf(right_identity,axiom,
    multiply(X,identity) = X ).

cnf(right_inverse,axiom,
    multiply(X,inverse(X)) = identity ).

cnf(squareness,hypothesis,
    multiply(X,X) = identity ).

cnf(a_times_b_is_c,hypothesis,
    multiply(a,b) = c ).

cnf(prove_b_times_a_is_c,negated_conjecture,
    multiply(b,a) != c ).

%------------------------------------------------------------------------------