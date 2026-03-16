% dynamiczna baza wiedzy dla silnii z 2 argumentami
% możemy wyświetlić bazę za pomocą `listing(s).`
:- dynamic s/2.

% silnia z 0 to 1
s(0, 1) :- !.

% silnia z liczby N to N pomnożone przez silnię z N-1
% wynik zapisujemy na początku bazy
s(N, X) :- N1 is N - 1, s(N1, X1), 
           X is N * X1, 
           asserta((s(N, X) :- !)).