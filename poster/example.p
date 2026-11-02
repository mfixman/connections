fof(leaders_on_two_legs_are_more_equal, axiom, ![X] : ((leader(X) & two_legs(X)) => more_equal(X))).
fof(heroes_are_leaders, axiom, ![X] : (hero(X) => leader(X))).
fof(napoleon_is_a_leader, axiom, leader(napoleon)).
fof(pigs_walk_on_two_legs, axiom, ![X] : (pig(X) => two_legs(X))).
fof(napoleon_is_a_pig, axiom, pig(napoleon)).
fof(boxer_is_a_hero, axiom, hero(boxer)).
fof(napoleon_is_more_equal, conjecture, more_equal(napoleon)).
