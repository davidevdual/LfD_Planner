begin_version
3
end_version
begin_metric
0
end_metric
9
begin_variable
var0
-1
2
Atom at(handright, gelatin_box)
NegatedAtom at(handright, gelatin_box)
end_variable
begin_variable
var1
-1
2
Atom enclosed(handright, gelatin_box)
NegatedAtom enclosed(handright, gelatin_box)
end_variable
begin_variable
var2
-1
2
Atom at(handright, bowl)
NegatedAtom at(handright, bowl)
end_variable
begin_variable
var3
-1
2
Atom enclosed(handright, bowl)
NegatedAtom enclosed(handright, bowl)
end_variable
begin_variable
var4
-1
2
Atom at(handleft, gelatin_box)
NegatedAtom at(handleft, gelatin_box)
end_variable
begin_variable
var5
-1
2
Atom enclosed(handleft, gelatin_box)
NegatedAtom enclosed(handleft, gelatin_box)
end_variable
begin_variable
var6
-1
2
Atom at(handleft, bowl)
NegatedAtom at(handleft, bowl)
end_variable
begin_variable
var7
-1
2
Atom enclosed(handleft, bowl)
NegatedAtom enclosed(handleft, bowl)
end_variable
begin_variable
var8
-1
2
Atom poured(gelatin_box, 9, bowl, 4)
NegatedAtom poured(gelatin_box, 9, bowl, 4)
end_variable
0
begin_state
1
1
1
1
1
1
1
1
1
end_state
begin_goal
1
8 0
end_goal
10
begin_operator
approach handleft bowl
0
1
0 6 1 0
1
end_operator
begin_operator
approach handleft gelatin_box
0
1
0 4 1 0
1
end_operator
begin_operator
approach handright bowl
0
1
0 2 1 0
1
end_operator
begin_operator
approach handright gelatin_box
0
1
0 0 1 0
1
end_operator
begin_operator
enclose handleft bowl
1
6 0
1
0 7 1 0
1
end_operator
begin_operator
enclose handleft gelatin_box
1
4 0
1
0 5 1 0
1
end_operator
begin_operator
enclose handright bowl
1
2 0
1
0 3 1 0
1
end_operator
begin_operator
enclose handright gelatin_box
1
0 0
1
0 1 1 0
1
end_operator
begin_operator
pour handleft handright gelatin_box 9 bowl 4
5
6 0
7 1
5 0
3 0
1 1
1
0 8 -1 0
1
end_operator
begin_operator
pour handright handleft gelatin_box 9 bowl 4
5
2 0
7 0
5 1
3 1
1 0
1
0 8 -1 0
1
end_operator
0
