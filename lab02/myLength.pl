% długość pustej listy to 0
myLength([], 0).

% długość listy to długość jej ogona powiększona o 1
myLength([_|T], X) :- myLength(T, X1), X is X1 + 1.