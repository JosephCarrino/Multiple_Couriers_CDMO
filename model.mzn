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
array[1..m, 1..n+2] of var int: c;


% The total distance travelled
var int: z;

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
    sum(j in 2..n+1 where c[i,j] != n+1)(s[c[i,j]]) <= l[i]
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

% Uncomment if couriers cannot go back to base before taking other packages
constraint forall(i in 1..m)(
    forall(j in 2..n+1)((c[i,j] == n+1) -> not exists(k in j..n+1)(c[i,k] != n+1))
);
  
  

% The distance to minimize is the max of the distances travelled by each
% courier
z = max(i in 1..m)(sum(j in 1..n+1)(D[c[i,j], c[i,j+1]]));
  
  
  

solve minimize z;

