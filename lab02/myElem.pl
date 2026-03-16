% element X jest w liście kiedy jest jej głową
myElem([X|_], X).

% element X jest w liście kiedy znajduje się w jej ogonie
myElem([_|T], X) :- myElem(T, X).
