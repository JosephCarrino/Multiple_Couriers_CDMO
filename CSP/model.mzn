% multiple couriers problem
include "globals.mzn";

% The number of couriers
int: m;

% The number of packages
int: n;

% The maximum weight a courier can carry
array[1..m] of int: l;

% The weight of each package
array[1..n] of int: s;

% The distance matrix
array[1..n+1, 1..n+1] of int: D;

% The matrix of integers indicating the chosen route by each courier
array[1..m, 1..n+2] of var int : c;


% The total distance travelled
var int: z;

% Array indicating how much weight the courier i transport
array[1..m] of var int: weights = [sum(j in 2..n+1 where c[i,j] != n+1)(s[c[i,j]]) | i in 1..m];

% Array indicating how much distance the courier i travel
array[1..m] of var int: distances = [sum(j in 1..n+1)(D[c[i,j], c[i,j+1]]) | i in 1..m];


% A package is carried by only one courier only one time
constraint
all_different(i in 1..m, j in 1..n+2 where c[i,j] != n+1)(c[i,j]);

% Each package is assigned to a courier
constraint forall(p in 1..n)(
    exists(i in 1..m, j in 1..n+2)(p == c[i,j])
);


% The total weight carried by each courier is less than or equal to his
% maximum carriable weight
constraint forall(i in 1..m)(
    weights[i] <= l[i]
    %sum(j in 2..n+1 where c[i,j] != n+1)(s[c[i,j]]) <= l[i]
);

% Every carried package must be delivered to destination 
% and every courier must start from destination
constraint forall(i in 1..m)(
    c[i,1] = n+1 /\ c[i, n+2] = n+1
);

% Couriers must immediatly start with a package after the base
constraint forall(i in 1..m)(
    exists(j in 1..n+1)(c[i,j] != n+1) -> (c[i, 2] != n+1)
);


% Comment if couriers can go back to base before taking other packages
constraint forall(i in 1..m)(
    forall(j in 2..n+1)((c[i,j] == n+1) -> not exists(k in j..n+1)(c[i,k] != n+1))
);

% If a courier can take a package, he will
constraint forall(c1 in 1..m)
  (
  (l[c1] >= max(s)) -> exists(k in 1..n+1)(c[c1,k] != n+1));


% Symmetry breaking 


% If two couriers have the same capacity then they are symmetric,
% to break the symmetry we impose an order (for the package they pick up) betweem them.
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m, c2 in 1..m where c1 < c2)
     ((l[c1] == l[c2]) -> (lex_less(row(c, c1), row(c, c2))))
);



% Two couriers path are exchangeable if the maximum weight of the two is less than the minimum loading capacity
% in that case we impose an ordering between them
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m, c2 in 1..m where c1 < c2)
    (
      (max(weights[c1], weights[c2]) <= min(l[c1], l[c2])) ->
      (lex_less(row(c, c1), row(c,c2))))
);



% The distance to minimize is the max of the distances travelled by each
% courier
z = max(i in 1..m)(distances[i]);




solve minimize z;

